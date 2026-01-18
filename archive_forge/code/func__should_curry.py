from functools import reduce, partial
import inspect
import sys
from operator import attrgetter, not_
from importlib import import_module
from types import MethodType
from .utils import no_default
from . import _signatures as _sigs
def _should_curry(self, args, kwargs, exc=None):
    func = self.func
    args = self.args + args
    if self.keywords:
        kwargs = dict(self.keywords, **kwargs)
    if self._sigspec is None:
        sigspec = self._sigspec = _sigs.signature_or_spec(func)
        self._has_unknown_args = has_varargs(func, sigspec) is not False
    else:
        sigspec = self._sigspec
    if is_partial_args(func, args, kwargs, sigspec) is False:
        return False
    elif self._has_unknown_args:
        return True
    elif not is_valid_args(func, args, kwargs, sigspec):
        return True
    else:
        return False