from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReclaimPolicyValueValuesEnum(_messages.Enum):
    """Optional. Whether the persistent disk should be deleted when the
    workstation is deleted. Valid values are `DELETE` and `RETAIN`. Defaults
    to `DELETE`.

    Values:
      RECLAIM_POLICY_UNSPECIFIED: Do not use.
      DELETE: Delete the persistent disk when deleting the workstation.
      RETAIN: Keep the persistent disk when deleting the workstation. An
        administrator must manually delete the disk.
    """
    RECLAIM_POLICY_UNSPECIFIED = 0
    DELETE = 1
    RETAIN = 2