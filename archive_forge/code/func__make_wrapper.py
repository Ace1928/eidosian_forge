from functools import wraps
from inspect import getmembers, isfunction
from webob import exc
from .compat import is_bound_method as ismethod
from .decorators import expose
from .util import _cfg, iscontroller
def _make_wrapper(f):
    """return a wrapped function with a copy of the _pecan context"""

    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    wrapper._pecan = f._pecan.copy()
    return wrapper