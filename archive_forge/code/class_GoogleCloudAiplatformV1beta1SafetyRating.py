from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SafetyRating(_messages.Message):
    """Safety rating corresponding to the generated content.

  Enums:
    CategoryValueValuesEnum: Output only. Harm category.
    ProbabilityValueValuesEnum: Output only. Harm probability levels in the
      content.
    SeverityValueValuesEnum: Output only. Harm severity levels in the content.

  Fields:
    blocked: Output only. Indicates whether the content was filtered out
      because of this rating.
    category: Output only. Harm category.
    probability: Output only. Harm probability levels in the content.
    probabilityScore: Output only. Harm probability score.
    severity: Output only. Harm severity levels in the content.
    severityScore: Output only. Harm severity score.
  """

    class CategoryValueValuesEnum(_messages.Enum):
        """Output only. Harm category.

    Values:
      HARM_CATEGORY_UNSPECIFIED: The harm category is unspecified.
      HARM_CATEGORY_HATE_SPEECH: The harm category is hate speech.
      HARM_CATEGORY_DANGEROUS_CONTENT: The harm category is dangerous content.
      HARM_CATEGORY_HARASSMENT: The harm category is harassment.
      HARM_CATEGORY_SEXUALLY_EXPLICIT: The harm category is sexually explicit
        content.
    """
        HARM_CATEGORY_UNSPECIFIED = 0
        HARM_CATEGORY_HATE_SPEECH = 1
        HARM_CATEGORY_DANGEROUS_CONTENT = 2
        HARM_CATEGORY_HARASSMENT = 3
        HARM_CATEGORY_SEXUALLY_EXPLICIT = 4

    class ProbabilityValueValuesEnum(_messages.Enum):
        """Output only. Harm probability levels in the content.

    Values:
      HARM_PROBABILITY_UNSPECIFIED: Harm probability unspecified.
      NEGLIGIBLE: Negligible level of harm.
      LOW: Low level of harm.
      MEDIUM: Medium level of harm.
      HIGH: High level of harm.
    """
        HARM_PROBABILITY_UNSPECIFIED = 0
        NEGLIGIBLE = 1
        LOW = 2
        MEDIUM = 3
        HIGH = 4

    class SeverityValueValuesEnum(_messages.Enum):
        """Output only. Harm severity levels in the content.

    Values:
      HARM_SEVERITY_UNSPECIFIED: Harm severity unspecified.
      HARM_SEVERITY_NEGLIGIBLE: Negligible level of harm severity.
      HARM_SEVERITY_LOW: Low level of harm severity.
      HARM_SEVERITY_MEDIUM: Medium level of harm severity.
      HARM_SEVERITY_HIGH: High level of harm severity.
    """
        HARM_SEVERITY_UNSPECIFIED = 0
        HARM_SEVERITY_NEGLIGIBLE = 1
        HARM_SEVERITY_LOW = 2
        HARM_SEVERITY_MEDIUM = 3
        HARM_SEVERITY_HIGH = 4
    blocked = _messages.BooleanField(1)
    category = _messages.EnumField('CategoryValueValuesEnum', 2)
    probability = _messages.EnumField('ProbabilityValueValuesEnum', 3)
    probabilityScore = _messages.FloatField(4, variant=_messages.Variant.FLOAT)
    severity = _messages.EnumField('SeverityValueValuesEnum', 5)
    severityScore = _messages.FloatField(6, variant=_messages.Variant.FLOAT)