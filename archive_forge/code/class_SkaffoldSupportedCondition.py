from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SkaffoldSupportedCondition(_messages.Message):
    """SkaffoldSupportedCondition contains information about when support for
  the release's version of Skaffold ends.

  Enums:
    SkaffoldSupportStateValueValuesEnum: The Skaffold support state for this
      release's version of Skaffold.

  Fields:
    maintenanceModeTime: The time at which this release's version of Skaffold
      will enter maintenance mode.
    skaffoldSupportState: The Skaffold support state for this release's
      version of Skaffold.
    status: True if the version of Skaffold used by this release is supported.
    supportExpirationTime: The time at which this release's version of
      Skaffold will no longer be supported.
  """

    class SkaffoldSupportStateValueValuesEnum(_messages.Enum):
        """The Skaffold support state for this release's version of Skaffold.

    Values:
      SKAFFOLD_SUPPORT_STATE_UNSPECIFIED: Default value. This value is unused.
      SKAFFOLD_SUPPORT_STATE_SUPPORTED: This Skaffold version is currently
        supported.
      SKAFFOLD_SUPPORT_STATE_MAINTENANCE_MODE: This Skaffold version is in
        maintenance mode.
      SKAFFOLD_SUPPORT_STATE_UNSUPPORTED: This Skaffold version is no longer
        supported.
    """
        SKAFFOLD_SUPPORT_STATE_UNSPECIFIED = 0
        SKAFFOLD_SUPPORT_STATE_SUPPORTED = 1
        SKAFFOLD_SUPPORT_STATE_MAINTENANCE_MODE = 2
        SKAFFOLD_SUPPORT_STATE_UNSUPPORTED = 3
    maintenanceModeTime = _messages.StringField(1)
    skaffoldSupportState = _messages.EnumField('SkaffoldSupportStateValueValuesEnum', 2)
    status = _messages.BooleanField(3)
    supportExpirationTime = _messages.StringField(4)