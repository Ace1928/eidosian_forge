import logging
import os
import sys
from llvmlite import ir
from llvmlite.binding import Linkage
from numba.pycc import llvm_types as lt
from numba.core.cgutils import create_constant_array
from numba.core.compiler import compile_extra, Flags
from numba.core.compiler_lock import global_compiler_lock
from numba.core.registry import cpu_target
from numba.core.runtime import nrtdynmod
from numba.core import cgutils
class _ModuleCompiler(object):
    """A base class to compile Python modules to a single shared library or
    extension module.

    :param export_entries: a list of ExportEntry instances.
    :param module_name: the name of the exported module.
    """
    method_def_ty = ir.LiteralStructType((lt._int8_star, lt._void_star, lt._int32, lt._int8_star))
    method_def_ptr = ir.PointerType(method_def_ty)
    env_def_ty = ir.LiteralStructType((lt._void_star, lt._int32, lt._void_star, lt._void_star, lt._int32))
    env_def_ptr = ir.PointerType(env_def_ty)

    def __init__(self, export_entries, module_name, use_nrt=False, **aot_options):
        self.module_name = module_name
        self.export_python_wrap = False
        self.dll_exports = []
        self.export_entries = export_entries
        self.external_init_function = None
        self.use_nrt = use_nrt
        self.typing_context = cpu_target.typing_context
        self.context = cpu_target.target_context.with_aot_codegen(self.module_name, **aot_options)

    def _mangle_method_symbol(self, func_name):
        return '._pycc_method_%s' % (func_name,)

    def _emit_python_wrapper(self, llvm_module):
        """Emit generated Python wrapper and extension module code.
        """
        raise NotImplementedError

    @global_compiler_lock
    def _cull_exports(self):
        """Read all the exported functions/modules in the translator
        environment, and join them into a single LLVM module.
        """
        self.exported_function_types = {}
        self.function_environments = {}
        self.environment_gvs = {}
        codegen = self.context.codegen()
        library = codegen.create_library(self.module_name)
        flags = Flags()
        flags.no_compile = True
        if not self.export_python_wrap:
            flags.no_cpython_wrapper = True
            flags.no_cfunc_wrapper = True
        if self.use_nrt:
            flags.nrt = True
            nrt_module, _ = nrtdynmod.create_nrt_module(self.context)
            library.add_ir_module(nrt_module)
        for entry in self.export_entries:
            cres = compile_extra(self.typing_context, self.context, entry.function, entry.signature.args, entry.signature.return_type, flags, locals={}, library=library)
            func_name = cres.fndesc.llvm_func_name
            llvm_func = cres.library.get_function(func_name)
            if self.export_python_wrap:
                llvm_func.linkage = 'internal'
                wrappername = cres.fndesc.llvm_cpython_wrapper_name
                wrapper = cres.library.get_function(wrappername)
                wrapper.name = self._mangle_method_symbol(entry.symbol)
                wrapper.linkage = 'external'
                fnty = cres.target_context.call_conv.get_function_type(cres.fndesc.restype, cres.fndesc.argtypes)
                self.exported_function_types[entry] = fnty
                self.function_environments[entry] = cres.environment
                self.environment_gvs[entry] = cres.fndesc.env_name
            else:
                llvm_func.name = entry.symbol
                self.dll_exports.append(entry.symbol)
        if self.export_python_wrap:
            wrapper_module = library.create_ir_module('wrapper')
            self._emit_python_wrapper(wrapper_module)
            library.add_ir_module(wrapper_module)
        library.finalize()
        for fn in library.get_defined_functions():
            if fn.name not in self.dll_exports:
                if fn.linkage in {Linkage.private, Linkage.internal}:
                    fn.visibility = 'default'
                else:
                    fn.visibility = 'hidden'
        return library

    def write_llvm_bitcode(self, output, wrap=False, **kws):
        self.export_python_wrap = wrap
        library = self._cull_exports()
        with open(output, 'wb') as fout:
            fout.write(library.emit_bitcode())

    def write_native_object(self, output, wrap=False, **kws):
        self.export_python_wrap = wrap
        library = self._cull_exports()
        with open(output, 'wb') as fout:
            fout.write(library.emit_native_object())

    def emit_type(self, tyobj):
        ret_val = str(tyobj)
        if 'int' in ret_val:
            if ret_val.endswith(('8', '16', '32', '64')):
                ret_val += '_t'
        return ret_val

    def emit_header(self, output):
        fname, ext = os.path.splitext(output)
        with open(fname + '.h', 'w') as fout:
            fout.write(get_header())
            fout.write('\n/* Prototypes */\n')
            for export_entry in self.export_entries:
                name = export_entry.symbol
                restype = self.emit_type(export_entry.signature.return_type)
                args = ', '.join((self.emit_type(argtype) for argtype in export_entry.signature.args))
                fout.write('extern %s %s(%s);\n' % (restype, name, args))

    def _emit_method_array(self, llvm_module):
        """
        Collect exported methods and emit a PyMethodDef array.

        :returns: a pointer to the PyMethodDef array.
        """
        method_defs = []
        for entry in self.export_entries:
            name = entry.symbol
            llvm_func_name = self._mangle_method_symbol(name)
            fnty = self.exported_function_types[entry]
            lfunc = ir.Function(llvm_module, fnty, llvm_func_name)
            method_name = self.context.insert_const_string(llvm_module, name)
            method_def_const = ir.Constant.literal_struct((method_name, ir.Constant.bitcast(lfunc, lt._void_star), METH_VARARGS_AND_KEYWORDS, NULL))
            method_defs.append(method_def_const)
        sentinel = ir.Constant.literal_struct([NULL, NULL, ZERO, NULL])
        method_defs.append(sentinel)
        method_array_init = create_constant_array(self.method_def_ty, method_defs)
        method_array = cgutils.add_global_variable(llvm_module, method_array_init.type, '.module_methods')
        method_array.initializer = method_array_init
        method_array.linkage = 'internal'
        method_array_ptr = ir.Constant.gep(method_array, [ZERO, ZERO])
        return method_array_ptr

    def _emit_environment_array(self, llvm_module, builder, pyapi):
        """
        Emit an array of env_def_t structures (see modulemixin.c)
        storing the pickled environment constants for each of the
        exported functions.
        """
        env_defs = []
        for entry in self.export_entries:
            env = self.function_environments[entry]
            env_def = pyapi.serialize_uncached(env.consts)
            env_defs.append(env_def)
        env_defs_init = create_constant_array(self.env_def_ty, env_defs)
        gv = self.context.insert_unique_const(llvm_module, '.module_environments', env_defs_init)
        return gv.gep([ZERO, ZERO])

    def _emit_envgvs_array(self, llvm_module, builder, pyapi):
        """
        Emit an array of Environment pointers that needs to be filled at
        initialization.
        """
        env_setters = []
        for entry in self.export_entries:
            envgv_name = self.environment_gvs[entry]
            gv = self.context.declare_env_global(llvm_module, envgv_name)
            envgv = gv.bitcast(lt._void_star)
            env_setters.append(envgv)
        env_setters_init = create_constant_array(lt._void_star, env_setters)
        gv = self.context.insert_unique_const(llvm_module, '.module_envgvs', env_setters_init)
        return gv.gep([ZERO, ZERO])

    def _emit_module_init_code(self, llvm_module, builder, modobj, method_array, env_array, envgv_array):
        """
        Emit call to "external" init function, if any.
        """
        if self.external_init_function:
            fnty = ir.FunctionType(lt._int32, [modobj.type, self.method_def_ptr, self.env_def_ptr, envgv_array.type])
            fn = ir.Function(llvm_module, fnty, self.external_init_function)
            return builder.call(fn, [modobj, method_array, env_array, envgv_array])
        else:
            return None