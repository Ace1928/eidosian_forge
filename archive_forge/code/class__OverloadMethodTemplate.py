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
class _OverloadMethodTemplate(_OverloadAttributeTemplate):
    """
    A base class of templates for @overload_method functions.
    """
    is_method = True

    def _init_once(self):
        """
        Overriding parent definition
        """
        attr = self._attr
        try:
            registry = self._get_target_registry('method')
        except InternalTargetMismatchError:
            pass
        else:
            lower_builtin = registry.lower

            @lower_builtin((self.key, attr), self.key, types.VarArg(types.Any))
            def method_impl(context, builder, sig, args):
                typ = sig.args[0]
                typing_context = context.typing_context
                fnty = self._get_function_type(typing_context, typ)
                sig = self._get_signature(typing_context, fnty, sig.args, {})
                call = context.get_function(fnty, sig)
                context.add_linking_libs(getattr(call, 'libs', ()))
                return call(builder, args)

    def _resolve(self, typ, attr):
        if self._attr != attr:
            return None
        if isinstance(typ, types.TypeRef):
            assert typ == self.key
        elif isinstance(typ, types.Callable):
            assert typ == self.key
        else:
            assert isinstance(typ, self.key)

        class MethodTemplate(AbstractTemplate):
            key = (self.key, attr)
            _inline = self._inline
            _overload_func = staticmethod(self._overload_func)
            _inline_overloads = self._inline_overloads
            prefer_literal = self.prefer_literal

            def generic(_, args, kws):
                args = (typ,) + tuple(args)
                fnty = self._get_function_type(self.context, typ)
                sig = self._get_signature(self.context, fnty, args, kws)
                sig = sig.replace(pysig=utils.pysignature(self._overload_func))
                for template in fnty.templates:
                    self._inline_overloads.update(template._inline_overloads)
                if sig is not None:
                    return sig.as_method()

            def get_template_info(self):
                basepath = os.path.dirname(os.path.dirname(numba.__file__))
                impl = self._overload_func
                code, firstlineno, path = self.get_source_code_info(impl)
                sig = str(utils.pysignature(impl))
                info = {'kind': 'overload_method', 'name': getattr(impl, '__qualname__', impl.__name__), 'sig': sig, 'filename': utils.safe_relpath(path, start=basepath), 'lines': (firstlineno, firstlineno + len(code) - 1), 'docstring': impl.__doc__}
                return info
        return types.BoundFunction(MethodTemplate, typ)