from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HttpMethodValueValuesEnum(_messages.Enum):
    """The HTTP method to use for the request. When specified, it overrides
    HttpRequest for the task. Note that if the value is set to HttpMethod the
    HttpRequest of the task will be ignored at execution time.

    Values:
      HTTP_METHOD_UNSPECIFIED: HTTP method unspecified
      POST: HTTP POST
      GET: HTTP GET
      HEAD: HTTP HEAD
      PUT: HTTP PUT
      DELETE: HTTP DELETE
      PATCH: HTTP PATCH
      OPTIONS: HTTP OPTIONS
    """
    HTTP_METHOD_UNSPECIFIED = 0
    POST = 1
    GET = 2
    HEAD = 3
    PUT = 4
    DELETE = 5
    PATCH = 6
    OPTIONS = 7