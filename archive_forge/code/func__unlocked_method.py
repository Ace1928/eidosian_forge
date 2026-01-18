from functools import wraps
from inspect import getmembers, isfunction
from webob import exc
from .compat import is_bound_method as ismethod
from .decorators import expose
from .util import _cfg, iscontroller
def _unlocked_method(func):
    _cfg(func)['secured'] = Any
    return func