from functools import wraps
from inspect import getmembers, isfunction
from webob import exc
from .compat import is_bound_method as ismethod
from .decorators import expose
from .util import _cfg, iscontroller
class _SecureState(object):

    def __init__(self, desc, boolean_value):
        self.description = desc
        self.boolean_value = boolean_value

    def __repr__(self):
        return '<SecureState %s>' % self.description

    def __nonzero__(self):
        return self.boolean_value

    def __bool__(self):
        return self.__nonzero__()