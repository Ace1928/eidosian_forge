from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2DataRiskLevel(_messages.Message):
    """Score is a summary of all elements in the data profile. A higher number
  means more risk.

  Enums:
    ScoreValueValuesEnum: The score applied to the resource.

  Fields:
    score: The score applied to the resource.
  """

    class ScoreValueValuesEnum(_messages.Enum):
        """The score applied to the resource.

    Values:
      RISK_SCORE_UNSPECIFIED: Unused.
      RISK_LOW: Low risk - Lower indication of sensitive data that appears to
        have additional access restrictions in place or no indication of
        sensitive data found.
      RISK_MODERATE: Medium risk - Sensitive data may be present but
        additional access or fine grain access restrictions appear to be
        present. Consider limiting access even further or transform data to
        mask.
      RISK_HIGH: High risk \\u2013 SPII may be present. Access controls may
        include public ACLs. Exfiltration of data may lead to user data loss.
        Re-identification of users may be possible. Consider limiting usage
        and or removing SPII.
    """
        RISK_SCORE_UNSPECIFIED = 0
        RISK_LOW = 1
        RISK_MODERATE = 2
        RISK_HIGH = 3
    score = _messages.EnumField('ScoreValueValuesEnum', 1)