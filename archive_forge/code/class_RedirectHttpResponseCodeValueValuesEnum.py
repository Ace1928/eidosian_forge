from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RedirectHttpResponseCodeValueValuesEnum(_messages.Enum):
    """30x code to use when performing redirects for the secure field.
    Defaults to 302.

    Values:
      REDIRECT_HTTP_RESPONSE_CODE_UNSPECIFIED: Not specified. 302 is assumed.
      REDIRECT_HTTP_RESPONSE_CODE_301: 301 Moved Permanently code.
      REDIRECT_HTTP_RESPONSE_CODE_302: 302 Moved Temporarily code.
      REDIRECT_HTTP_RESPONSE_CODE_303: 303 See Other code.
      REDIRECT_HTTP_RESPONSE_CODE_307: 307 Temporary Redirect code.
    """
    REDIRECT_HTTP_RESPONSE_CODE_UNSPECIFIED = 0
    REDIRECT_HTTP_RESPONSE_CODE_301 = 1
    REDIRECT_HTTP_RESPONSE_CODE_302 = 2
    REDIRECT_HTTP_RESPONSE_CODE_303 = 3
    REDIRECT_HTTP_RESPONSE_CODE_307 = 4