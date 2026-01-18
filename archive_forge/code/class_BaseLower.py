from collections import namedtuple, defaultdict
import operator
import warnings
from functools import partial
import llvmlite.ir
from llvmlite.ir import Constant, IRBuilder
from numba.core import (typing, utils, types, ir, debuginfo, funcdesc,
from numba.core.errors import (LoweringError, new_error_context, TypingError,
from numba.core.funcdesc import default_mangler
from numba.core.environment import Environment
from numba.core.analysis import compute_use_defs, must_use_alloca
from numba.misc.firstlinefinder import get_func_body_first_lineno
class BaseLower(object):
    """
    Lower IR to LLVM
    """

    def __init__(self, context, library, fndesc, func_ir, metadata=None):
        self.library = library
        self.fndesc = fndesc
        self.blocks = utils.SortedMap(func_ir.blocks.items())
        self.func_ir = func_ir
        self.generator_info = func_ir.generator_info
        self.metadata = metadata
        self.flags = targetconfig.ConfigStack.top_or_none()
        self.module = self.library.create_ir_module(self.fndesc.unique_name)
        self.env = Environment.from_fndesc(self.fndesc)
        self.blkmap = {}
        self.pending_phis = {}
        self.varmap = {}
        self.firstblk = min(self.blocks.keys())
        self.loc = -1
        self.context = context.subtarget(environment=self.env, fndesc=self.fndesc)
        dibuildercls = self.context.DIBuilder if self.context.enable_debuginfo else debuginfo.DummyDIBuilder
        self.defn_loc = self._compute_def_location()
        directives_only = self.flags.dbg_directives_only
        self.debuginfo = dibuildercls(module=self.module, filepath=func_ir.loc.filename, cgctx=context, directives_only=directives_only)
        self.init()

    @property
    def call_conv(self):
        return self.context.call_conv

    def init(self):
        pass

    def init_pyapi(self):
        """
        Init the Python API and Environment Manager for the function being
        lowered.
        """
        if self.pyapi is not None:
            return
        self.pyapi = self.context.get_python_api(self.builder)
        self.env_manager = self.context.get_env_manager(self.builder)
        self.env_body = self.env_manager.env_body
        self.envarg = self.env_manager.env_ptr

    def _compute_def_location(self):
        defn_loc = self.func_ir.loc.with_lineno(self.func_ir.loc.line + 1)
        if self.context.enable_debuginfo:
            fn = self.func_ir.func_id.func
            optional_lno = get_func_body_first_lineno(fn)
            if optional_lno is not None:
                offset = optional_lno - 1
                defn_loc = self.func_ir.loc.with_lineno(offset)
            else:
                msg = f'Could not find source for function: {self.func_ir.func_id.func}. Debug line information may be inaccurate.'
                warnings.warn(NumbaDebugInfoWarning(msg))
        return defn_loc

    def pre_lower(self):
        """
        Called before lowering all blocks.
        """
        self.pyapi = None
        self.debuginfo.mark_subprogram(function=self.builder.function, qualname=self.fndesc.qualname, argnames=self.fndesc.args, argtypes=self.fndesc.argtypes, line=self.defn_loc.line)
        attributes = self.builder.function.attributes
        full_debug = self.flags.debuginfo and (not self.flags.dbg_directives_only)
        if full_debug and 'alwaysinline' not in attributes:
            attributes.add('noinline')

    def post_lower(self):
        """
        Called after all blocks are lowered
        """
        self.debuginfo.finalize()

    def pre_block(self, block):
        """
        Called before lowering a block.
        """

    def post_block(self, block):
        """
        Called after lowering a block.
        """

    def return_dynamic_exception(self, exc_class, exc_args, nb_types, loc=None):
        self.call_conv.return_dynamic_user_exc(self.builder, exc_class, exc_args, nb_types, loc=loc, func_name=self.func_ir.func_id.func_name)

    def return_exception(self, exc_class, exc_args=None, loc=None):
        """Propagate exception to the caller.
        """
        self.call_conv.return_user_exc(self.builder, exc_class, exc_args, loc=loc, func_name=self.func_ir.func_id.func_name)

    def set_exception(self, exc_class, exc_args=None, loc=None):
        """Set exception state in the current function.
        """
        self.call_conv.set_static_user_exc(self.builder, exc_class, exc_args, loc=loc, func_name=self.func_ir.func_id.func_name)

    def emit_environment_object(self):
        """Emit a pointer to hold the Environment object.
        """
        envname = self.context.get_env_name(self.fndesc)
        self.context.declare_env_global(self.module, envname)

    def lower(self):
        self.emit_environment_object()
        if self.generator_info is None:
            self.genlower = None
            self.lower_normal_function(self.fndesc)
        else:
            self.genlower = self.GeneratorLower(self)
            self.gentype = self.genlower.gentype
            self.genlower.lower_init_func(self)
            self.genlower.lower_next_func(self)
            if self.gentype.has_finalizer:
                self.genlower.lower_finalize_func(self)
        if config.DUMP_LLVM:
            utils.dump_llvm(self.fndesc, self.module)
        if self.context.enable_nrt and self.generator_info is None:
            removerefctpass.remove_unnecessary_nrt_usage(self.function, context=self.context, fndesc=self.fndesc)
        self.context.post_lowering(self.module, self.library)
        self.library.add_ir_module(self.module)

    def extract_function_arguments(self):
        self.fnargs = self.call_conv.decode_arguments(self.builder, self.fndesc.argtypes, self.function)
        return self.fnargs

    def lower_normal_function(self, fndesc):
        """
        Lower non-generator *fndesc*.
        """
        self.setup_function(fndesc)
        self.extract_function_arguments()
        entry_block_tail = self.lower_function_body()
        with debuginfo.suspend_emission(self.builder):
            self.builder.position_at_end(entry_block_tail)
            self.builder.branch(self.blkmap[self.firstblk])

    def lower_function_body(self):
        """
        Lower the current function's body, and return the entry block.
        """
        for offset in self.blocks:
            bname = 'B%s' % offset
            self.blkmap[offset] = self.function.append_basic_block(bname)
        self.pre_lower()
        entry_block_tail = self.builder.basic_block
        self.debug_print('# function begin: {0}'.format(self.fndesc.unique_name))
        for offset, block in sorted(self.blocks.items()):
            bb = self.blkmap[offset]
            self.builder.position_at_end(bb)
            self.debug_print(f'# lower block: {offset}')
            self.lower_block(block)
        self.post_lower()
        return entry_block_tail

    def lower_block(self, block):
        """
        Lower the given block.
        """
        self.pre_block(block)
        for inst in block.body:
            self.loc = inst.loc
            defaulterrcls = partial(LoweringError, loc=self.loc)
            with new_error_context('lowering "{inst}" at {loc}', inst=inst, loc=self.loc, errcls_=defaulterrcls):
                self.lower_inst(inst)
        self.post_block(block)

    def create_cpython_wrapper(self, release_gil=False):
        """
        Create CPython wrapper(s) around this function (or generator).
        """
        if self.genlower:
            self.context.create_cpython_wrapper(self.library, self.genlower.gendesc, self.env, self.call_helper, release_gil=release_gil)
        self.context.create_cpython_wrapper(self.library, self.fndesc, self.env, self.call_helper, release_gil=release_gil)

    def create_cfunc_wrapper(self):
        """
        Create C wrapper around this function.
        """
        if self.genlower:
            raise UnsupportedError('generator as a first-class function type')
        self.context.create_cfunc_wrapper(self.library, self.fndesc, self.env, self.call_helper)

    def setup_function(self, fndesc):
        self.function = self.context.declare_function(self.module, fndesc)
        if self.flags.dbg_optnone:
            attrset = self.function.attributes
            if 'alwaysinline' not in attrset:
                attrset.add('optnone')
                attrset.add('noinline')
        self.entry_block = self.function.append_basic_block('entry')
        self.builder = IRBuilder(self.entry_block)
        self.call_helper = self.call_conv.init_call_helper(self.builder)

    def typeof(self, varname):
        return self.fndesc.typemap[varname]

    def debug_print(self, msg):
        if config.DEBUG_JIT:
            self.context.debug_print(self.builder, f'DEBUGJIT [{self.fndesc.qualname}]: {msg}')

    def print_variable(self, msg, varname):
        """Helper to emit ``print(msg, varname)`` for debugging.

        Parameters
        ----------
        msg : str
            Literal string to be printed.
        varname : str
            A variable name whose value will be printed.
        """
        argtys = (types.literal(msg), self.fndesc.typemap[varname])
        args = (self.context.get_dummy_value(), self.loadvar(varname))
        sig = typing.signature(types.none, *argtys)
        impl = self.context.get_function(print, sig)
        impl(self.builder, args)