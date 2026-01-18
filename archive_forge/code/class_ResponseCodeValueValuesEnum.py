from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResponseCodeValueValuesEnum(_messages.Enum):
    """The HTTP Status code to use for the redirect.

    Values:
      RESPONSE_CODE_UNSPECIFIED: Default value
      MOVED_PERMANENTLY_DEFAULT: Corresponds to 301.
      FOUND: Corresponds to 302.
      SEE_OTHER: Corresponds to 303.
      TEMPORARY_REDIRECT: Corresponds to 307. In this case, the request method
        will be retained.
      PERMANENT_REDIRECT: Corresponds to 308. In this case, the request method
        will be retained.
    """
    RESPONSE_CODE_UNSPECIFIED = 0
    MOVED_PERMANENTLY_DEFAULT = 1
    FOUND = 2
    SEE_OTHER = 3
    TEMPORARY_REDIRECT = 4
    PERMANENT_REDIRECT = 5