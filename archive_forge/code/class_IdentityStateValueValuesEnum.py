from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IdentityStateValueValuesEnum(_messages.Enum):
    """Output only. The current state of an identity set in policy.

    Values:
      IDENTITY_STATE_UNSPECIFIED: Not used
      ACTIVE: Identity is active.
      DELETED: Identity is deleted.
    """
    IDENTITY_STATE_UNSPECIFIED = 0
    ACTIVE = 1
    DELETED = 2