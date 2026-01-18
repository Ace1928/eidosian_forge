from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KeyRevocationActionTypeValueValuesEnum(_messages.Enum):
    """Optional. KeyRevocationActionType of the instance.

    Values:
      KEY_REVOCATION_ACTION_TYPE_UNSPECIFIED: Default value. This value is
        unused.
      NONE: Indicates user chose no operation.
      STOP: Indicates user chose to opt for VM shutdown on key revocation.
    """
    KEY_REVOCATION_ACTION_TYPE_UNSPECIFIED = 0
    NONE = 1
    STOP = 2