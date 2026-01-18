from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RequestMethodValueValuesEnum(_messages.Enum):
    """The HTTP request method to use for the check. If set to
    METHOD_UNSPECIFIED then request_method defaults to GET.

    Values:
      METHOD_UNSPECIFIED: No request method specified.
      GET: GET request.
      POST: POST request.
    """
    METHOD_UNSPECIFIED = 0
    GET = 1
    POST = 2