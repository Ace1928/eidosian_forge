from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecurityHealthAnalyticsModule(_messages.Message):
    """Message for Security Health Analytics built-in detector.

  Enums:
    ModuleEnablementStateValueValuesEnum: The state of enablement for the
      module at its level of the resource hierarchy.

  Fields:
    moduleEnablementState: The state of enablement for the module at its level
      of the resource hierarchy.
    moduleName: Required. The name of the module eg:
      BIGQUERY_TABLE_CMEK_DISABLED.
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
    moduleEnablementState = _messages.EnumField('ModuleEnablementStateValueValuesEnum', 1)
    moduleName = _messages.StringField(2)