from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Violation(_messages.Message):
    """Details of violation.

  Enums:
    SeverityValueValuesEnum: Severity of the violation.

  Fields:
    assetId: Asset which violated some policy.
    nextSteps: Next steps or recommendations to act upon this violation.
    policyId: Policy which was violated by the asset.
    severity: Severity of the violation.
    violatedAsset: Details of the asset which got violated.
    violatedPolicy: Details of the policy which got violated.
    violatedPosture: Posture details if the violated policy belongs to a
      posture deployment.
  """

    class SeverityValueValuesEnum(_messages.Enum):
        """Severity of the violation.

    Values:
      SEVERITY_UNSPECIFIED: This is the default severity if the severity is
        unknown.
      CRITICAL: <no description>
      HIGH: <no description>
      MEDIUM: <no description>
      LOW: <no description>
    """
        SEVERITY_UNSPECIFIED = 0
        CRITICAL = 1
        HIGH = 2
        MEDIUM = 3
        LOW = 4
    assetId = _messages.StringField(1)
    nextSteps = _messages.StringField(2)
    policyId = _messages.StringField(3)
    severity = _messages.EnumField('SeverityValueValuesEnum', 4)
    violatedAsset = _messages.MessageField('AssetDetails', 5)
    violatedPolicy = _messages.MessageField('PolicyDetails', 6)
    violatedPosture = _messages.MessageField('PostureDetails', 7)