from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HttpRouteCorsPolicy(_messages.Message):
    """The Specification for allowing client side cross-origin requests.

  Fields:
    allowCredentials: In response to a preflight request, setting this to true
      indicates that the actual request can include user credentials. This
      translates to the Access-Control-Allow-Credentials header. Default value
      is false.
    allowHeaders: Specifies the content for Access-Control-Allow-Headers
      header.
    allowMethods: Specifies the content for Access-Control-Allow-Methods
      header.
    allowOriginRegexes: Specifies the regular expression patterns that match
      allowed origins. For regular expression grammar, please see
      https://github.com/google/re2/wiki/Syntax.
    allowOrigins: Specifies the list of origins that will be allowed to do
      CORS requests. An origin is allowed if it matches either an item in
      allow_origins or an item in allow_origin_regexes.
    disabled: If true, the CORS policy is disabled. The default value is
      false, which indicates that the CORS policy is in effect.
    exposeHeaders: Specifies the content for Access-Control-Expose-Headers
      header.
    maxAge: Specifies how long result of a preflight request can be cached in
      seconds. This translates to the Access-Control-Max-Age header.
  """
    allowCredentials = _messages.BooleanField(1)
    allowHeaders = _messages.StringField(2, repeated=True)
    allowMethods = _messages.StringField(3, repeated=True)
    allowOriginRegexes = _messages.StringField(4, repeated=True)
    allowOrigins = _messages.StringField(5, repeated=True)
    disabled = _messages.BooleanField(6)
    exposeHeaders = _messages.StringField(7, repeated=True)
    maxAge = _messages.StringField(8)