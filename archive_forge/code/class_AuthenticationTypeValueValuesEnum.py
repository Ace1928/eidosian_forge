from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AuthenticationTypeValueValuesEnum(_messages.Enum):
    """Authentication type for workload execution.

    Values:
      AUTHENTICATION_TYPE_UNSPECIFIED: If AuthenticationType is unspecified
        then SERVICE_ACCOUNT is used
      SERVICE_ACCOUNT: Use service account credentials for authentication
      CREDENTIALS_INJECTION: Use injected credentials for authentication
      END_USER_CREDENTIALS: Use end user credentials for authentication
        (go/dataproc-personal-auth-v2)
    """
    AUTHENTICATION_TYPE_UNSPECIFIED = 0
    SERVICE_ACCOUNT = 1
    CREDENTIALS_INJECTION = 2
    END_USER_CREDENTIALS = 3