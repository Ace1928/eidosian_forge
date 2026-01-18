from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsCloudbuildV2Condition(_messages.Message):
    """Conditions defines a readiness condition for a Knative resource.

  Enums:
    SeverityValueValuesEnum: Severity with which to treat failures of this
      type of condition.
    StatusValueValuesEnum: Status of the condition.

  Fields:
    lastTransitionTime: LastTransitionTime is the last time the condition
      transitioned from one status to another.
    message: A human readable message indicating details about the transition.
    reason: The reason for the condition's last transition.
    severity: Severity with which to treat failures of this type of condition.
    status: Status of the condition.
    type: Type of condition.
  """

    class SeverityValueValuesEnum(_messages.Enum):
        """Severity with which to treat failures of this type of condition.

    Values:
      SEVERITY_UNSPECIFIED: Default enum type; should not be used.
      WARNING: Severity is warning.
      INFO: Severity is informational only.
    """
        SEVERITY_UNSPECIFIED = 0
        WARNING = 1
        INFO = 2

    class StatusValueValuesEnum(_messages.Enum):
        """Status of the condition.

    Values:
      UNKNOWN: Default enum type indicating execution is still ongoing.
      TRUE: Success
      FALSE: Failure
    """
        UNKNOWN = 0
        TRUE = 1
        FALSE = 2
    lastTransitionTime = _messages.StringField(1)
    message = _messages.StringField(2)
    reason = _messages.StringField(3)
    severity = _messages.EnumField('SeverityValueValuesEnum', 4)
    status = _messages.EnumField('StatusValueValuesEnum', 5)
    type = _messages.StringField(6)