from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RedirectResponseCodeValueValuesEnum(_messages.Enum):
    """Optional. The HTTP status code to use for this redirect action. For a
    list of supported values, see RedirectResponseCode.

    Values:
      MOVED_PERMANENTLY_DEFAULT: `HTTP 301 (Moved Permanently)`
      FOUND: HTTP 302 Found
      SEE_OTHER: HTTP 303 See Other
      TEMPORARY_REDIRECT: `HTTP 307 (Temporary Redirect)`. In this case, the
        request method is retained.
      PERMANENT_REDIRECT: `HTTP 308 (Permanent Redirect)`. In this case, the
        request method is retained.
    """
    MOVED_PERMANENTLY_DEFAULT = 0
    FOUND = 1
    SEE_OTHER = 2
    TEMPORARY_REDIRECT = 3
    PERMANENT_REDIRECT = 4