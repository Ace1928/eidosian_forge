from functools import wraps
from inspect import getmembers, isfunction
from webob import exc
from .compat import is_bound_method as ismethod
from .decorators import expose
from .util import _cfg, iscontroller
def _allowed_check_permissions_types(x):
    return ismethod(x) or isfunction(x) or isinstance(x, str)