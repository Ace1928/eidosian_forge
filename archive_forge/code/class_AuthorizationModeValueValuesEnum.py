from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AuthorizationModeValueValuesEnum(_messages.Enum):
    """Optional. The authorization mode of the Redis cluster. If not
    provided, auth feature is disabled for the cluster.

    Values:
      AUTH_MODE_UNSPECIFIED: Not set.
      AUTH_MODE_IAM_AUTH: IAM basic authorization mode
      AUTH_MODE_DISABLED: Authorization disabled mode
    """
    AUTH_MODE_UNSPECIFIED = 0
    AUTH_MODE_IAM_AUTH = 1
    AUTH_MODE_DISABLED = 2