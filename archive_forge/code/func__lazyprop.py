import functools
from typing import Callable, TypeVar
from cvxpy.utilities import scopes
@property
@functools.wraps(func)
def _lazyprop(self):
    if scopes.dpp_scope_active():
        attr_name = '_lazy_dpp_' + func.__name__
    else:
        attr_name = '_lazy_' + func.__name__
    try:
        return getattr(self, attr_name)
    except AttributeError:
        setattr(self, attr_name, func(self))
    return getattr(self, attr_name)