from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HttpElementLocationValueValuesEnum(_messages.Enum):
    """Required. The location of the API key.

    Values:
      HTTP_IN_UNSPECIFIED: <no description>
      HTTP_IN_QUERY: Element is in the HTTP request query.
      HTTP_IN_HEADER: Element is in the HTTP request header.
      HTTP_IN_PATH: Element is in the HTTP request path.
      HTTP_IN_BODY: Element is in the HTTP request body.
      HTTP_IN_COOKIE: Element is in the HTTP request cookie.
    """
    HTTP_IN_UNSPECIFIED = 0
    HTTP_IN_QUERY = 1
    HTTP_IN_HEADER = 2
    HTTP_IN_PATH = 3
    HTTP_IN_BODY = 4
    HTTP_IN_COOKIE = 5