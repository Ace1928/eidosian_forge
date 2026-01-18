from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SystemParameter(_messages.Message):
    """Define a parameter's name and location. The parameter may be passed as
  either an HTTP header or a URL query parameter, and if both are passed the
  behavior is implementation-dependent.

  Fields:
    httpHeader: Define the HTTP header name to use for the parameter. It is
      case insensitive.
    name: Define the name of the parameter, such as "api_key", "alt",
      "callback", and etc. It is case sensitive.
    urlQueryParameter: Define the URL query parameter name to use for the
      parameter. It is case sensitive.
  """
    httpHeader = _messages.StringField(1)
    name = _messages.StringField(2)
    urlQueryParameter = _messages.StringField(3)