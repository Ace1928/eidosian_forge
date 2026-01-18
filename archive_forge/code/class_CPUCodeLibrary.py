import warnings
import functools
import locale
import weakref
import ctypes
import html
import textwrap
import llvmlite.binding as ll
import llvmlite.ir as llvmir
from abc import abstractmethod, ABCMeta
from numba.core import utils, config, cgutils
from numba.core.llvm_bindings import create_pass_manager_builder
from numba.core.runtime.nrtopt import remove_redundant_nrt_refct
from numba.core.runtime import rtsys
from numba.core.compiler_lock import require_global_compiler_lock
from numba.core.errors import NumbaInvalidConfigWarning
from numba.misc.inspection import disassemble_elf_to_cfg
from numba.misc.llvm_pass_timings import PassTimingsCollection
class CPUCodeLibrary(CodeLibrary):

    def __init__(self, codegen, name):
        super().__init__(codegen, name)
        self._linking_libraries = []
        self._final_module = ll.parse_assembly(str(self._codegen._create_empty_module(self.name)))
        self._final_module.name = cgutils.normalize_ir_text(self.name)
        self._shared_module = None

    def _optimize_functions(self, ll_module):
        """
        Internal: run function-level optimizations inside *ll_module*.
        """
        ll_module.data_layout = self._codegen._data_layout
        with self._codegen._function_pass_manager(ll_module) as fpm:
            for func in ll_module.functions:
                k = f'Function passes on {func.name!r}'
                with self._recorded_timings.record(k):
                    fpm.initialize()
                    fpm.run(func)
                    fpm.finalize()

    def _optimize_final_module(self):
        """
        Internal: optimize this library's final module.
        """
        cheap_name = 'Module passes (cheap optimization for refprune)'
        with self._recorded_timings.record(cheap_name):
            self._codegen._mpm_cheap.run(self._final_module)
        if not config.LLVM_REFPRUNE_PASS:
            self._final_module = remove_redundant_nrt_refct(self._final_module)
        full_name = 'Module passes (full optimization)'
        with self._recorded_timings.record(full_name):
            self._codegen._mpm_full.run(self._final_module)

    def _get_module_for_linking(self):
        """
        Internal: get a LLVM module suitable for linking multiple times
        into another library.  Exported functions are made "linkonce_odr"
        to allow for multiple definitions, inlining, and removal of
        unused exports.

        See discussion in https://github.com/numba/numba/pull/890
        """
        self._ensure_finalized()
        if self._shared_module is not None:
            return self._shared_module
        mod = self._final_module
        to_fix = []
        nfuncs = 0
        for fn in mod.functions:
            nfuncs += 1
            if not fn.is_declaration and fn.linkage == ll.Linkage.external:
                to_fix.append(fn.name)
        if nfuncs == 0:
            raise RuntimeError('library unfit for linking: no available functions in %s' % (self,))
        if to_fix:
            mod = mod.clone()
            for name in to_fix:
                mod.get_function(name).linkage = 'linkonce_odr'
        self._shared_module = mod
        return mod

    def add_linking_library(self, library):
        library._ensure_finalized()
        self._linking_libraries.append(library)

    def add_ir_module(self, ir_module):
        self._raise_if_finalized()
        assert isinstance(ir_module, llvmir.Module)
        ir = cgutils.normalize_ir_text(str(ir_module))
        ll_module = ll.parse_assembly(ir)
        ll_module.name = ir_module.name
        ll_module.verify()
        self.add_llvm_module(ll_module)

    def add_llvm_module(self, ll_module):
        self._optimize_functions(ll_module)
        if not config.LLVM_REFPRUNE_PASS:
            ll_module = remove_redundant_nrt_refct(ll_module)
        self._final_module.link_in(ll_module)

    def finalize(self):
        require_global_compiler_lock()
        self._codegen._check_llvm_bugs()
        self._raise_if_finalized()
        if config.DUMP_FUNC_OPT:
            dump('FUNCTION OPTIMIZED DUMP %s' % self.name, self.get_llvm_str(), 'llvm')
        seen = set()
        for library in self._linking_libraries:
            if library not in seen:
                seen.add(library)
                self._final_module.link_in(library._get_module_for_linking(), preserve=True)
        self._optimize_final_module()
        self._final_module.verify()
        self._finalize_final_module()

    def _finalize_dynamic_globals(self):
        for gv in self._final_module.global_variables:
            if gv.name.startswith('numba.dynamic.globals'):
                self._dynamic_globals.append(gv.name)

    def _verify_declare_only_symbols(self):
        for fn in self._final_module.functions:
            if fn.is_declaration and fn.name.startswith('_ZN5numba'):
                msg = 'Symbol {} not linked properly'
                raise AssertionError(msg.format(fn.name))

    def _finalize_final_module(self):
        """
        Make the underlying LLVM module ready to use.
        """
        self._finalize_dynamic_globals()
        self._verify_declare_only_symbols()
        self._final_module.__library = weakref.proxy(self)
        cleanup = self._codegen._add_module(self._final_module)
        if cleanup:
            weakref.finalize(self, cleanup)
        self._finalize_specific()
        self._finalized = True
        if config.DUMP_OPTIMIZED:
            dump('OPTIMIZED DUMP %s' % self.name, self.get_llvm_str(), 'llvm')
        if config.DUMP_ASSEMBLY:
            dump('ASSEMBLY %s' % self.name, self.get_asm_str(), 'asm')

    def get_defined_functions(self):
        """
        Get all functions defined in the library.  The library must have
        been finalized.
        """
        mod = self._final_module
        for fn in mod.functions:
            if not fn.is_declaration:
                yield fn

    def get_function(self, name):
        return self._final_module.get_function(name)

    def _sentry_cache_disable_inspection(self):
        if self._disable_inspection:
            warnings.warn('Inspection disabled for cached code. Invalid result is returned.')

    def get_llvm_str(self):
        self._sentry_cache_disable_inspection()
        return str(self._final_module)

    def get_asm_str(self):
        self._sentry_cache_disable_inspection()
        return str(self._codegen._tm.emit_assembly(self._final_module))

    def get_function_cfg(self, name, py_func=None, **kwargs):
        """
        Get control-flow graph of the LLVM function
        """
        self._sentry_cache_disable_inspection()
        return _CFG(self, name, py_func, **kwargs)

    def get_disasm_cfg(self, mangled_name):
        """
        Get the CFG of the disassembly of the ELF object at symbol mangled_name.

        Requires python package: r2pipe
        Requires radare2 binary on $PATH.
        Notebook rendering requires python package: graphviz
        Optionally requires a compiler toolchain (via pycc) to link the ELF to
        get better disassembly results.
        """
        elf = self._get_compiled_object()
        return disassemble_elf_to_cfg(elf, mangled_name)

    @classmethod
    def _dump_elf(cls, buf):
        """
        Dump the symbol table of an ELF file.
        Needs pyelftools (https://github.com/eliben/pyelftools)
        """
        from elftools.elf.elffile import ELFFile
        from elftools.elf import descriptions
        from io import BytesIO
        f = ELFFile(BytesIO(buf))
        print('ELF file:')
        for sec in f.iter_sections():
            if sec['sh_type'] == 'SHT_SYMTAB':
                symbols = sorted(sec.iter_symbols(), key=lambda sym: sym.name)
                print('    symbols:')
                for sym in symbols:
                    if not sym.name:
                        continue
                    print('    - %r: size=%d, value=0x%x, type=%s, bind=%s' % (sym.name.decode(), sym['st_size'], sym['st_value'], descriptions.describe_symbol_type(sym['st_info']['type']), descriptions.describe_symbol_bind(sym['st_info']['bind'])))
        print()

    @classmethod
    def _object_compiled_hook(cls, ll_module, buf):
        """
        `ll_module` was compiled into object code `buf`.
        """
        try:
            self = ll_module.__library
        except AttributeError:
            return
        if self._object_caching_enabled:
            self._compiled = True
            self._compiled_object = buf

    @classmethod
    def _object_getbuffer_hook(cls, ll_module):
        """
        Return a cached object code for `ll_module`.
        """
        try:
            self = ll_module.__library
        except AttributeError:
            return
        if self._object_caching_enabled and self._compiled_object:
            buf = self._compiled_object
            self._compiled_object = None
            return buf

    def serialize_using_bitcode(self):
        """
        Serialize this library using its bitcode as the cached representation.
        """
        self._ensure_finalized()
        return (self.name, 'bitcode', self._final_module.as_bitcode())

    def serialize_using_object_code(self):
        """
        Serialize this library using its object code as the cached
        representation.  We also include its bitcode for further inlining
        with other libraries.
        """
        self._ensure_finalized()
        data = (self._get_compiled_object(), self._get_module_for_linking().as_bitcode())
        return (self.name, 'object', data)

    @classmethod
    def _unserialize(cls, codegen, state):
        name, kind, data = state
        self = codegen.create_library(name)
        assert isinstance(self, cls)
        if kind == 'bitcode':
            self._final_module = ll.parse_bitcode(data)
            self._finalize_final_module()
            return self
        elif kind == 'object':
            object_code, shared_bitcode = data
            self.enable_object_caching()
            self._set_compiled_object(object_code)
            self._shared_module = ll.parse_bitcode(shared_bitcode)
            self._finalize_final_module()
            self._codegen._engine._load_defined_symbols(self._shared_module)
            return self
        else:
            raise ValueError('unsupported serialization kind %r' % (kind,))