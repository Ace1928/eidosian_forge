from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.core.util import files
def SetAuthorizationNamePath(unused_ref, unused_args, request):
    """Sets the request path in name attribute for authorization request.

  Appends /authorization at the end of the request path for the singleton
  authorization request.

  Args:
    unused_ref: reference to the project object.
    unused_args: command line arguments.
    request: API request to be issued

  Returns:
    modified request
  """
    del unused_ref, unused_args
    request.name = request.name + '/authorization'
    return request