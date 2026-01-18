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
class CallableTemplate(FunctionTemplate):
    """
    Base class for a template defining a ``generic(self)`` method
    returning a callable to be called with the actual ``*args`` and
    ``**kwargs`` representing the call signature.  The callable has
    to return a return type, a full signature, or None.  The signature
    does not have to match the input types. It is compared against the
    input types afterwards.
    """
    recvr = None

    def apply(self, args, kws):
        generic = getattr(self, 'generic')
        typer = generic()
        match_sig = inspect.signature(typer)
        try:
            match_sig.bind(*args, **kws)
        except TypeError as e:
            raise TypingError(str(e)) from e
        sig = typer(*args, **kws)
        if sig is None:
            if any((isinstance(x, types.Optional) for x in args)):

                def unpack_opt(x):
                    if isinstance(x, types.Optional):
                        return x.type
                    else:
                        return x
                args = list(map(unpack_opt, args))
                sig = typer(*args, **kws)
            if sig is None:
                return
        try:
            pysig = typer.pysig
        except AttributeError:
            pysig = utils.pysignature(typer)
        bound = pysig.bind(*args, **kws)
        if bound.kwargs:
            raise TypingError('unsupported call signature')
        if not isinstance(sig, Signature):
            if not isinstance(sig, types.Type):
                raise TypeError('invalid return type for callable template: got %r' % (sig,))
            sig = signature(sig, *bound.args)
        if self.recvr is not None:
            sig = sig.replace(recvr=self.recvr)
        if len(bound.args) < len(pysig.parameters):
            parameters = list(pysig.parameters.values())[:len(bound.args)]
            pysig = pysig.replace(parameters=parameters)
        sig = sig.replace(pysig=pysig)
        cases = [sig]
        return self._select(cases, bound.args, bound.kwargs)

    def get_template_info(self):
        impl = getattr(self, 'generic')
        basepath = os.path.dirname(os.path.dirname(numba.__file__))
        code, firstlineno, path = self.get_source_code_info(impl)
        sig = str(utils.pysignature(impl))
        info = {'kind': 'overload', 'name': getattr(self.key, '__name__', getattr(impl, '__qualname__', impl.__name__)), 'sig': sig, 'filename': utils.safe_relpath(path, start=basepath), 'lines': (firstlineno, firstlineno + len(code) - 1), 'docstring': impl.__doc__}
        return info