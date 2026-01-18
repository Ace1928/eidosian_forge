from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ValidationCheck(_messages.Message):
    """ValidationCheck represents the result of preflight check.

  Enums:
    OptionValueValuesEnum: Options used for the validation check
    ScenarioValueValuesEnum: Output only. The scenario when the preflight
      checks were run.

  Fields:
    option: Options used for the validation check
    scenario: Output only. The scenario when the preflight checks were run.
    status: Output only. The detailed validation check status.
  """

    class OptionValueValuesEnum(_messages.Enum):
        """Options used for the validation check

    Values:
      OPTIONS_UNSPECIFIED: Default value. Standard preflight validation check
        will be used.
      SKIP_VALIDATION_CHECK_BLOCKING: Prevent failed preflight checks from
        failing.
      SKIP_VALIDATION_ALL: Skip all preflight check validations.
    """
        OPTIONS_UNSPECIFIED = 0
        SKIP_VALIDATION_CHECK_BLOCKING = 1
        SKIP_VALIDATION_ALL = 2

    class ScenarioValueValuesEnum(_messages.Enum):
        """Output only. The scenario when the preflight checks were run.

    Values:
      SCENARIO_UNSPECIFIED: Default value. This value is unused.
      CREATE: The validation check occurred during a create flow.
      UPDATE: The validation check occurred during an update flow.
    """
        SCENARIO_UNSPECIFIED = 0
        CREATE = 1
        UPDATE = 2
    option = _messages.EnumField('OptionValueValuesEnum', 1)
    scenario = _messages.EnumField('ScenarioValueValuesEnum', 2)
    status = _messages.MessageField('ValidationCheckStatus', 3)