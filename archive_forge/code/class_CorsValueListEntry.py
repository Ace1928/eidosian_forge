from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CorsValueListEntry(_messages.Message):
    """A CorsValueListEntry object.

    Fields:
      maxAgeSeconds: The value, in seconds, to return in the  Access-Control-
        Max-Age header used in preflight responses.
      method: The list of HTTP methods on which to include CORS response
        headers, (GET, OPTIONS, POST, etc) Note: "*" is permitted in the list
        of methods, and means "any method".
      origin: The list of Origins eligible to receive CORS response headers.
        Note: "*" is permitted in the list of origins, and means "any Origin".
      responseHeader: The list of HTTP headers other than the simple response
        headers to give permission for the user-agent to share across domains.
    """
    maxAgeSeconds = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    method = _messages.StringField(2, repeated=True)
    origin = _messages.StringField(3, repeated=True)
    responseHeader = _messages.StringField(4, repeated=True)