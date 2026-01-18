from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecurityLevelValueValuesEnum(_messages.Enum):
    """Security (HTTPS) enforcement for this URL.

    Values:
      SECURE_UNSPECIFIED: Not specified.
      SECURE_DEFAULT: Both HTTP and HTTPS requests with URLs that match the
        handler succeed without redirects. The application can examine the
        request to determine which protocol was used, and respond accordingly.
      SECURE_NEVER: Requests for a URL that match this handler that use HTTPS
        are automatically redirected to the HTTP equivalent URL.
      SECURE_OPTIONAL: Both HTTP and HTTPS requests with URLs that match the
        handler succeed without redirects. The application can examine the
        request to determine which protocol was used and respond accordingly.
      SECURE_ALWAYS: Requests for a URL that match this handler that do not
        use HTTPS are automatically redirected to the HTTPS URL with the same
        path. Query parameters are reserved for the redirect.
    """
    SECURE_UNSPECIFIED = 0
    SECURE_DEFAULT = 1
    SECURE_NEVER = 2
    SECURE_OPTIONAL = 3
    SECURE_ALWAYS = 4