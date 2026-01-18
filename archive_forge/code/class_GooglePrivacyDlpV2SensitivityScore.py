from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2SensitivityScore(_messages.Message):
    """Score is calculated from of all elements in the data profile. A higher
  level means the data is more sensitive.

  Enums:
    ScoreValueValuesEnum: The sensitivity score applied to the resource.

  Fields:
    score: The sensitivity score applied to the resource.
  """

    class ScoreValueValuesEnum(_messages.Enum):
        """The sensitivity score applied to the resource.

    Values:
      SENSITIVITY_SCORE_UNSPECIFIED: Unused.
      SENSITIVITY_LOW: No sensitive information detected. The resource isn't
        publicly accessible.
      SENSITIVITY_MODERATE: Medium risk. Contains personally identifiable
        information (PII), potentially sensitive data, or fields with free-
        text data that are at a higher risk of having intermittent sensitive
        data. Consider limiting access.
      SENSITIVITY_HIGH: High risk. Sensitive personally identifiable
        information (SPII) can be present. Exfiltration of data can lead to
        user data loss. Re-identification of users might be possible. Consider
        limiting usage and or removing SPII.
    """
        SENSITIVITY_SCORE_UNSPECIFIED = 0
        SENSITIVITY_LOW = 1
        SENSITIVITY_MODERATE = 2
        SENSITIVITY_HIGH = 3
    score = _messages.EnumField('ScoreValueValuesEnum', 1)