from inspect import ismethod, getmembers
import warnings
from webob import exc
from .core import abort
from .decorators import expose
from .routing import lookup_controller, handle_lookup_traversal
from .util import iscontroller, getargspec
def _get_args_for_controller(self, controller):
    """
        Retrieve the arguments we actually care about.  For Pecan applications
        that utilize thread locals, we should truncate the first argument,
        `self`.  For applications that explicitly pass request/response
        references as the first controller arguments, we should truncate the
        first three arguments, `self, req, resp`.
        """
    argspec = getargspec(controller)
    from pecan import request
    try:
        request.path
    except AttributeError:
        return argspec.args[3:]
    return argspec.args[1:]