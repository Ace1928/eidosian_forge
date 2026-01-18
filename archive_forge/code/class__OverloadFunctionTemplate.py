from abc import ABC, abstractmethod
import functools
import sys
import inspect
import os.path
from collections import namedtuple
from collections.abc import Sequence
from types import MethodType, FunctionType, MappingProxyType
import numba
from numba.core import types, utils, targetconfig
from numba.core.errors import (
from numba.core.cpu_options import InlineOptions
class _OverloadFunctionTemplate(AbstractTemplate):
    """
    A base class of templates for overload functions.
    """

    def _validate_sigs(self, typing_func, impl_func):
        typing_sig = utils.pysignature(typing_func)
        impl_sig = utils.pysignature(impl_func)

        def get_args_kwargs(sig):
            kws = []
            args = []
            pos_arg = None
            for x in sig.parameters.values():
                if x.default == utils.pyParameter.empty:
                    args.append(x)
                    if x.kind == utils.pyParameter.VAR_POSITIONAL:
                        pos_arg = x
                    elif x.kind == utils.pyParameter.VAR_KEYWORD:
                        msg = "The use of VAR_KEYWORD (e.g. **kwargs) is unsupported. (offending argument name is '%s')"
                        raise InternalError(msg % x)
                else:
                    kws.append(x)
            return (args, kws, pos_arg)
        ty_args, ty_kws, ty_pos = get_args_kwargs(typing_sig)
        im_args, im_kws, im_pos = get_args_kwargs(impl_sig)
        sig_fmt = 'Typing signature:         %s\nImplementation signature: %s'
        sig_str = sig_fmt % (typing_sig, impl_sig)
        err_prefix = 'Typing and implementation arguments differ in '
        a = ty_args
        b = im_args
        if ty_pos:
            if not im_pos:
                msg = "VAR_POSITIONAL (e.g. *args) argument kind (offending argument name is '%s') found in the typing function signature, but is not in the implementing function signature.\n%s" % (ty_pos, sig_str)
                raise InternalError(msg)
        elif im_pos:
            b = im_args[:im_args.index(im_pos)]
            try:
                a = ty_args[:ty_args.index(b[-1]) + 1]
            except ValueError:
                specialized = "argument names.\n%s\nFirst difference: '%s'"
                msg = err_prefix + specialized % (sig_str, b[-1])
                raise InternalError(msg)

        def gen_diff(typing, implementing):
            diff = set(typing) ^ set(implementing)
            return 'Difference: %s' % diff
        if a != b:
            specialized = 'argument names.\n%s\n%s' % (sig_str, gen_diff(a, b))
            raise InternalError(err_prefix + specialized)
        ty = [x.name for x in ty_kws]
        im = [x.name for x in im_kws]
        if ty != im:
            specialized = 'keyword argument names.\n%s\n%s'
            msg = err_prefix + specialized % (sig_str, gen_diff(ty_kws, im_kws))
            raise InternalError(msg)
        same = [x.default for x in ty_kws] == [x.default for x in im_kws]
        if not same:
            specialized = 'keyword argument default values.\n%s\n%s'
            msg = err_prefix + specialized % (sig_str, gen_diff(ty_kws, im_kws))
            raise InternalError(msg)

    def generic(self, args, kws):
        """
        Type the overloaded function by compiling the appropriate
        implementation for the given args.
        """
        from numba.core.typed_passes import PreLowerStripPhis
        disp, new_args = self._get_impl(args, kws)
        if disp is None:
            return
        disp_type = types.Dispatcher(disp)
        if not self._inline.is_never_inline:
            from numba.core import typed_passes, compiler
            from numba.core.inline_closurecall import InlineWorker
            fcomp = disp._compiler
            flags = compiler.Flags()
            tyctx = fcomp.targetdescr.typing_context
            tgctx = fcomp.targetdescr.target_context
            compiler_inst = fcomp.pipeline_class(tyctx, tgctx, None, None, None, flags, None)
            inline_worker = InlineWorker(tyctx, tgctx, fcomp.locals, compiler_inst, flags, None)
            resolve = disp_type.dispatcher.get_call_template
            template, pysig, folded_args, kws = resolve(new_args, kws)
            ir = inline_worker.run_untyped_passes(disp_type.dispatcher.py_func, enable_ssa=True)
            typemap, return_type, calltypes, _ = typed_passes.type_inference_stage(self.context, tgctx, ir, folded_args, None)
            ir = PreLowerStripPhis()._strip_phi_nodes(ir)
            ir._definitions = numba.core.ir_utils.build_definitions(ir.blocks)
            sig = Signature(return_type, folded_args, None)
            self._inline_overloads[sig.args] = {'folded_args': folded_args}
            impl_init = _EmptyImplementationEntry('always inlined')
            self._compiled_overloads[sig.args] = impl_init
            if not self._inline.is_always_inline:
                sig = disp_type.get_call_type(self.context, new_args, kws)
                self._compiled_overloads[sig.args] = disp_type.get_overload(sig)
            iinfo = _inline_info(ir, typemap, calltypes, sig)
            self._inline_overloads[sig.args] = {'folded_args': folded_args, 'iinfo': iinfo}
        else:
            sig = disp_type.get_call_type(self.context, new_args, kws)
            if sig is None:
                return None
            self._compiled_overloads[sig.args] = disp_type.get_overload(sig)
        return sig

    def _get_impl(self, args, kws):
        """Get implementation given the argument types.

        Returning a Dispatcher object.  The Dispatcher object is cached
        internally in `self._impl_cache`.
        """
        flags = targetconfig.ConfigStack.top_or_none()
        cache_key = (self.context, tuple(args), tuple(kws.items()), flags)
        try:
            impl, args = self._impl_cache[cache_key]
            return (impl, args)
        except KeyError:
            pass
        impl, args = self._build_impl(cache_key, args, kws)
        return (impl, args)

    def _get_jit_decorator(self):
        """Gets a jit decorator suitable for the current target"""
        from numba.core.target_extension import target_registry, get_local_target, jit_registry
        jitter_str = self.metadata.get('target', 'generic')
        jitter = jit_registry.get(jitter_str, None)
        if jitter is None:
            target_class = target_registry.get(jitter_str, None)
            if target_class is None:
                msg = ("Unknown target '{}', has it been ", 'registered?')
                raise ValueError(msg.format(jitter_str))
            target_hw = get_local_target(self.context)
            if not issubclass(target_hw, target_class):
                msg = 'No overloads exist for the requested target: {}.'
            jitter = jit_registry[target_hw]
        if jitter is None:
            raise ValueError('Cannot find a suitable jit decorator')
        return jitter

    def _build_impl(self, cache_key, args, kws):
        """Build and cache the implementation.

        Given the positional (`args`) and keyword arguments (`kws`), obtains
        the `overload` implementation and wrap it in a Dispatcher object.
        The expected argument types are returned for use by type-inference.
        The expected argument types are only different from the given argument
        types if there is an imprecise type in the given argument types.

        Parameters
        ----------
        cache_key : hashable
            The key used for caching the implementation.
        args : Tuple[Type]
            Types of positional argument.
        kws : Dict[Type]
            Types of keyword argument.

        Returns
        -------
        disp, args :
            On success, returns `(Dispatcher, Tuple[Type])`.
            On failure, returns `(None, None)`.

        """
        jitter = self._get_jit_decorator()
        ov_sig = inspect.signature(self._overload_func)
        try:
            ov_sig.bind(*args, **kws)
        except TypeError as e:
            raise TypingError(str(e)) from e
        else:
            ovf_result = self._overload_func(*args, **kws)
        if ovf_result is None:
            self._impl_cache[cache_key] = (None, None)
            return (None, None)
        elif isinstance(ovf_result, tuple):
            sig, pyfunc = ovf_result
            args = sig.args
            kws = {}
            cache_key = None
        else:
            pyfunc = ovf_result
        if not isinstance(pyfunc, FunctionType):
            msg = 'Implementation function returned by `@overload` has an unexpected type.  Got {}'
            raise AssertionError(msg.format(pyfunc))
        if self._strict:
            self._validate_sigs(self._overload_func, pyfunc)
        jitdecor = jitter(**self._jit_options)
        disp = jitdecor(pyfunc)
        disp_type = types.Dispatcher(disp)
        disp_type.get_call_type(self.context, args, kws)
        if cache_key is not None:
            self._impl_cache[cache_key] = (disp, args)
        return (disp, args)

    def get_impl_key(self, sig):
        """
        Return the key for looking up the implementation for the given
        signature on the target context.
        """
        return self._compiled_overloads[sig.args]

    @classmethod
    def get_source_info(cls):
        """Return a dictionary with information about the source code of the
        implementation.

        Returns
        -------
        info : dict
            - "kind" : str
                The implementation kind.
            - "name" : str
                The name of the function that provided the definition.
            - "sig" : str
                The formatted signature of the function.
            - "filename" : str
                The name of the source file.
            - "lines": tuple (int, int)
                First and list line number.
            - "docstring": str
                The docstring of the definition.
        """
        basepath = os.path.dirname(os.path.dirname(numba.__file__))
        impl = cls._overload_func
        code, firstlineno, path = cls.get_source_code_info(impl)
        sig = str(utils.pysignature(impl))
        info = {'kind': 'overload', 'name': getattr(impl, '__qualname__', impl.__name__), 'sig': sig, 'filename': utils.safe_relpath(path, start=basepath), 'lines': (firstlineno, firstlineno + len(code) - 1), 'docstring': impl.__doc__}
        return info

    def get_template_info(self):
        basepath = os.path.dirname(os.path.dirname(numba.__file__))
        impl = self._overload_func
        code, firstlineno, path = self.get_source_code_info(impl)
        sig = str(utils.pysignature(impl))
        info = {'kind': 'overload', 'name': getattr(impl, '__qualname__', impl.__name__), 'sig': sig, 'filename': utils.safe_relpath(path, start=basepath), 'lines': (firstlineno, firstlineno + len(code) - 1), 'docstring': impl.__doc__}
        return info