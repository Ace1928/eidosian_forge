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
class AbstractTemplate(FunctionTemplate):
    """
    Defines method ``generic(self, args, kws)`` which compute a possible
    signature base on input types.  The signature does not have to match the
    input types. It is compared against the input types afterwards.
    """

    def apply(self, args, kws):
        generic = getattr(self, 'generic')
        sig = generic(args, kws)
        if sig is not None:
            if not isinstance(sig, Signature):
                raise AssertionError('generic() must return a Signature or None. {} returned {}'.format(generic, type(sig)))
        if not sig and any((isinstance(x, types.Optional) for x in args)):

            def unpack_opt(x):
                if isinstance(x, types.Optional):
                    return x.type
                else:
                    return x
            args = list(map(unpack_opt, args))
            assert not kws
            sig = generic(args, kws)
        return sig

    def get_template_info(self):
        impl = getattr(self, 'generic')
        basepath = os.path.dirname(os.path.dirname(numba.__file__))
        code, firstlineno, path = self.get_source_code_info(impl)
        sig = str(utils.pysignature(impl))
        info = {'kind': 'overload', 'name': getattr(impl, '__qualname__', impl.__name__), 'sig': sig, 'filename': utils.safe_relpath(path, start=basepath), 'lines': (firstlineno, firstlineno + len(code) - 1), 'docstring': impl.__doc__}
        return info