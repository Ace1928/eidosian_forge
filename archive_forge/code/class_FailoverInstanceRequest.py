from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FailoverInstanceRequest(_messages.Message):
    """Request for Failover.

  Enums:
    DataProtectionModeValueValuesEnum: Optional. Available data protection
      modes that the user can choose. If it's unspecified, data protection
      mode will be LIMITED_DATA_LOSS by default.

  Fields:
    dataProtectionMode: Optional. Available data protection modes that the
      user can choose. If it's unspecified, data protection mode will be
      LIMITED_DATA_LOSS by default.
  """

    class DataProtectionModeValueValuesEnum(_messages.Enum):
        """Optional. Available data protection modes that the user can choose. If
    it's unspecified, data protection mode will be LIMITED_DATA_LOSS by
    default.

    Values:
      DATA_PROTECTION_MODE_UNSPECIFIED: Defaults to LIMITED_DATA_LOSS if a
        data protection mode is not specified.
      LIMITED_DATA_LOSS: Instance failover will be protected with data loss
        control. More specifically, the failover will only be performed if the
        current replication offset diff between primary and replica is under a
        certain threshold.
      FORCE_DATA_LOSS: Instance failover will be performed without data loss
        control.
    """
        DATA_PROTECTION_MODE_UNSPECIFIED = 0
        LIMITED_DATA_LOSS = 1
        FORCE_DATA_LOSS = 2
    dataProtectionMode = _messages.EnumField('DataProtectionModeValueValuesEnum', 1)