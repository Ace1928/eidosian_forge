from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpgradeHistoryEntry(_messages.Message):
    """The entry of VM image upgrade history.

  Enums:
    ActionValueValuesEnum: Optional. Action. Rolloback or Upgrade.
    StateValueValuesEnum: Output only. The state of this instance upgrade
      history entry.

  Fields:
    action: Optional. Action. Rolloback or Upgrade.
    containerImage: Optional. The container image before this instance
      upgrade.
    createTime: Immutable. The time that this instance upgrade history entry
      is created.
    framework: Optional. The framework of this notebook instance.
    snapshot: Optional. The snapshot of the boot disk of this notebook
      instance before upgrade.
    state: Output only. The state of this instance upgrade history entry.
    targetVersion: Optional. Target VM Version, like m63.
    version: Optional. The version of the notebook instance before this
      upgrade.
    vmImage: Optional. The VM image before this instance upgrade.
  """

    class ActionValueValuesEnum(_messages.Enum):
        """Optional. Action. Rolloback or Upgrade.

    Values:
      ACTION_UNSPECIFIED: Operation is not specified.
      UPGRADE: Upgrade.
      ROLLBACK: Rollback.
    """
        ACTION_UNSPECIFIED = 0
        UPGRADE = 1
        ROLLBACK = 2

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The state of this instance upgrade history entry.

    Values:
      STATE_UNSPECIFIED: State is not specified.
      STARTED: The instance upgrade is started.
      SUCCEEDED: The instance upgrade is succeeded.
      FAILED: The instance upgrade is failed.
    """
        STATE_UNSPECIFIED = 0
        STARTED = 1
        SUCCEEDED = 2
        FAILED = 3
    action = _messages.EnumField('ActionValueValuesEnum', 1)
    containerImage = _messages.StringField(2)
    createTime = _messages.StringField(3)
    framework = _messages.StringField(4)
    snapshot = _messages.StringField(5)
    state = _messages.EnumField('StateValueValuesEnum', 6)
    targetVersion = _messages.StringField(7)
    version = _messages.StringField(8)
    vmImage = _messages.StringField(9)