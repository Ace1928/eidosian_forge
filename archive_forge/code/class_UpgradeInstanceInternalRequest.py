from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpgradeInstanceInternalRequest(_messages.Message):
    """Request for upgrading a notebook instance from within the VM

  Enums:
    TypeValueValuesEnum: Optional. The optional UpgradeType. Setting this
      field will search for additional compute images to upgrade this
      instance.

  Fields:
    type: Optional. The optional UpgradeType. Setting this field will search
      for additional compute images to upgrade this instance.
    vmId: Required. The VM hardware token for authenticating the VM.
      https://cloud.google.com/compute/docs/instances/verifying-instance-
      identity
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Optional. The optional UpgradeType. Setting this field will search for
    additional compute images to upgrade this instance.

    Values:
      UPGRADE_TYPE_UNSPECIFIED: Upgrade type is not specified.
      UPGRADE_FRAMEWORK: Upgrade ML framework.
      UPGRADE_OS: Upgrade Operating System.
      UPGRADE_CUDA: Upgrade CUDA.
      UPGRADE_ALL: Upgrade All (OS, Framework and CUDA).
    """
        UPGRADE_TYPE_UNSPECIFIED = 0
        UPGRADE_FRAMEWORK = 1
        UPGRADE_OS = 2
        UPGRADE_CUDA = 3
        UPGRADE_ALL = 4
    type = _messages.EnumField('TypeValueValuesEnum', 1)
    vmId = _messages.StringField(2)