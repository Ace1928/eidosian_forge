from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AuthorizationHeaderValueValuesEnum(_messages.Enum):
    """Configuration for the token sent in the `HTTP` Authorization header.

    Values:
      HTTP_AUTHORIZATION_HEADER_UNSPECIFIED: Default value, equivalent to
        `SYSTEM_ID_TOKEN`.
      SYSTEM_ID_TOKEN: Send an ID token for the project-specific Google
        Workspace Add-on's system service account (default).
      USER_ID_TOKEN: Send an ID token for the end user.
      NONE: Do not send an Authentication header.
    """
    HTTP_AUTHORIZATION_HEADER_UNSPECIFIED = 0
    SYSTEM_ID_TOKEN = 1
    USER_ID_TOKEN = 2
    NONE = 3