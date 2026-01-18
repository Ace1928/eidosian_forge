from inspect import ismethod, getmembers
import warnings
from webob import exc
from .core import abort
from .decorators import expose
from .routing import lookup_controller, handle_lookup_traversal
from .util import iscontroller, getargspec
def _handle_bad_rest_arguments(self, controller, remainder, request):
    """
        Ensure that the argspec for a discovered controller actually matched
        the positional arguments in the request path.  If not, raise
        a webob.exc.HTTPBadRequest.
        """
    argspec = self._get_args_for_controller(controller)
    fixed_args = len(argspec) - len(request.pecan.get('routing_args', []))
    if len(remainder) < fixed_args:
        abort(404)