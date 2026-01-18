from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecurityHealthAnalyticsCustomModule(_messages.Message):
    """Message for SHA Custom Module

  Enums:
    ModuleEnablementStateValueValuesEnum: The state of enablement for the
      module at its level of the resource hierarchy.

  Fields:
    config: Required. custom module details
    displayName: Optional. The display name of the Security Health Analytics
      custom module. This display name becomes the finding category for all
      findings that are returned by this custom module. The display name must
      be between 1 and 128 characters, start with a lowercase letter, and
      contain alphanumeric characters or underscores only.
    id: Output only. Immutable. The id of the custom module. The id is server-
      generated and is not user settable. It will be a numeric id containing
      1-20 digits.
    moduleEnablementState: The state of enablement for the module at its level
      of the resource hierarchy.
  """

    class ModuleEnablementStateValueValuesEnum(_messages.Enum):
        """The state of enablement for the module at its level of the resource
    hierarchy.

    Values:
      ENABLEMENT_STATE_UNSPECIFIED: Default value. This value is unused.
      ENABLED: State is enabled.
      DISABLED: State is disabled.
    """
        ENABLEMENT_STATE_UNSPECIFIED = 0
        ENABLED = 1
        DISABLED = 2
    config = _messages.MessageField('CustomConfig', 1)
    displayName = _messages.StringField(2)
    id = _messages.StringField(3)
    moduleEnablementState = _messages.EnumField('ModuleEnablementStateValueValuesEnum', 4)