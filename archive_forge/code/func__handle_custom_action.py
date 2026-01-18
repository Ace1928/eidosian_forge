from inspect import ismethod, getmembers
import warnings
from webob import exc
from .core import abort
from .decorators import expose
from .routing import lookup_controller, handle_lookup_traversal
from .util import iscontroller, getargspec
def _handle_custom_action(self, method, remainder, request=None):
    if request is None:
        self._raise_method_deprecation_warning(self._handle_custom_action)
    remainder = [r for r in remainder if r]
    if remainder:
        if method in ('put', 'delete'):
            method_name = remainder[0]
            remainder = remainder[1:]
        else:
            method_name = remainder[-1]
            remainder = remainder[:-1]
        if method.upper() in self._custom_actions.get(method_name, []):
            controller = self._find_controller('%s_%s' % (method, method_name), method_name)
            if controller:
                return (controller, remainder)