from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HttpCacheControlResponseHeader(_messages.Message):
    """RFC-2616: cache control support

  Fields:
    age: 14.6 response cache age, in seconds since the response is generated
    directive: 14.9 request and response directives
    expires: 14.21 response cache expires, in RFC 1123 date format
  """
    age = _messages.IntegerField(1)
    directive = _messages.StringField(2)
    expires = _messages.StringField(3)