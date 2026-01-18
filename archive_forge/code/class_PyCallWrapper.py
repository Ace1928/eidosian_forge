from llvmlite.ir import Constant, IRBuilder
import llvmlite.ir
from numba.core import types, config, cgutils
class PyCallWrapper(object):

    def __init__(self, context, module, func, fndesc, env, call_helper, release_gil):
        self.context = context
        self.module = module
        self.func = func
        self.fndesc = fndesc
        self.env = env
        self.release_gil = release_gil

    def build(self):
        wrapname = self.fndesc.llvm_cpython_wrapper_name
        pyobj = self.context.get_argument_type(types.pyobject)
        wrapty = llvmlite.ir.FunctionType(pyobj, [pyobj, pyobj, pyobj])
        wrapper = llvmlite.ir.Function(self.module, wrapty, name=wrapname)
        builder = IRBuilder(wrapper.append_basic_block('entry'))
        closure, args, kws = wrapper.args
        closure.name = 'py_closure'
        args.name = 'py_args'
        kws.name = 'py_kws'
        api = self.context.get_python_api(builder)
        self.build_wrapper(api, builder, closure, args, kws)
        return (wrapper, api)

    def build_wrapper(self, api, builder, closure, args, kws):
        nargs = len(self.fndesc.argtypes)
        objs = [api.alloca_obj() for _ in range(nargs)]
        parseok = api.unpack_tuple(args, self.fndesc.qualname, nargs, nargs, *objs)
        pred = builder.icmp_unsigned('==', parseok, Constant(parseok.type, None))
        with cgutils.if_unlikely(builder, pred):
            builder.ret(api.get_null_object())
        endblk = builder.append_basic_block('arg.end')
        with builder.goto_block(endblk):
            builder.ret(api.get_null_object())
        env_manager = self.get_env(api, builder)
        cleanup_manager = _ArgManager(self.context, builder, api, env_manager, endblk, nargs)
        innerargs = []
        for obj, ty in zip(objs, self.fndesc.argtypes):
            if isinstance(ty, types.Omitted):
                innerargs.append(None)
            else:
                val = cleanup_manager.add_arg(builder.load(obj), ty)
                innerargs.append(val)
        if self.release_gil:
            cleanup_manager = _GilManager(builder, api, cleanup_manager)
        status, retval = self.context.call_conv.call_function(builder, self.func, self.fndesc.restype, self.fndesc.argtypes, innerargs, attrs=('noinline',))
        self.debug_print(builder, '# callwrapper: emit_cleanup')
        cleanup_manager.emit_cleanup()
        self.debug_print(builder, '# callwrapper: emit_cleanup end')
        with builder.if_then(status.is_ok, likely=True):
            with builder.if_then(status.is_none):
                api.return_none()
            retty = self._simplified_return_type()
            obj = api.from_native_return(retty, retval, env_manager)
            builder.ret(obj)
        self.context.call_conv.raise_error(builder, api, status)
        builder.ret(api.get_null_object())

    def get_env(self, api, builder):
        """Get the Environment object which is declared as a global
        in the module of the wrapped function.
        """
        envname = self.context.get_env_name(self.fndesc)
        gvptr = self.context.declare_env_global(builder.module, envname)
        envptr = builder.load(gvptr)
        env_body = self.context.get_env_body(builder, envptr)
        api.emit_environment_sentry(envptr, return_pyobject=True, debug_msg=self.fndesc.env_name)
        env_manager = api.get_env_manager(self.env, env_body, envptr)
        return env_manager

    def _simplified_return_type(self):
        """
        The NPM callconv has already converted simplified optional types.
        We can simply use the value type from it.
        """
        restype = self.fndesc.restype
        if isinstance(restype, types.Optional):
            return restype.type
        else:
            return restype

    def debug_print(self, builder, msg):
        if config.DEBUG_JIT:
            self.context.debug_print(builder, 'DEBUGJIT: {0}'.format(msg))