from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AutopilotCompatibilityIssue(_messages.Message):
    """AutopilotCompatibilityIssue contains information about a specific
  compatibility issue with Autopilot mode.

  Enums:
    IncompatibilityTypeValueValuesEnum: The incompatibility type of this
      issue.

  Fields:
    constraintType: The constraint type of the issue.
    description: The description of the issue.
    documentationUrl: A URL to a public documnetation, which addresses
      resolving this issue.
    incompatibilityType: The incompatibility type of this issue.
    lastObservation: The last time when this issue was observed.
    subjects: The name of the resources which are subject to this issue.
  """

    class IncompatibilityTypeValueValuesEnum(_messages.Enum):
        """The incompatibility type of this issue.

    Values:
      UNSPECIFIED: Default value, should not be used.
      INCOMPATIBILITY: Indicates that the issue is a known incompatibility
        between the cluster and Autopilot mode.
      ADDITIONAL_CONFIG_REQUIRED: Indicates the issue is an incompatibility if
        customers take no further action to resolve.
      PASSED_WITH_OPTIONAL_CONFIG: Indicates the issue is not an
        incompatibility, but depending on the workloads business logic, there
        is a potential that they won't work on Autopilot.
    """
        UNSPECIFIED = 0
        INCOMPATIBILITY = 1
        ADDITIONAL_CONFIG_REQUIRED = 2
        PASSED_WITH_OPTIONAL_CONFIG = 3
    constraintType = _messages.StringField(1)
    description = _messages.StringField(2)
    documentationUrl = _messages.StringField(3)
    incompatibilityType = _messages.EnumField('IncompatibilityTypeValueValuesEnum', 4)
    lastObservation = _messages.StringField(5)
    subjects = _messages.StringField(6, repeated=True)