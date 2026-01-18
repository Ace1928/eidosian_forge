from inspect import ismethod, getmembers
import warnings
from webob import exc
from .core import abort
from .decorators import expose
from .routing import lookup_controller, handle_lookup_traversal
from .util import iscontroller, getargspec
def _set_routing_args(self, request, args):
    """
        Sets default routing arguments.
        """
    request.pecan.setdefault('routing_args', []).extend(args)