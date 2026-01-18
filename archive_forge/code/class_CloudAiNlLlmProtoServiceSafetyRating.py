from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudAiNlLlmProtoServiceSafetyRating(_messages.Message):
    """Safety rating corresponding to the generated content.

  Enums:
    CategoryValueValuesEnum: Harm category.
    ProbabilityValueValuesEnum: Harm probability levels in the content.
    SeverityValueValuesEnum: Harm severity levels in the content.

  Fields:
    blocked: Indicates whether the content was filtered out because of this
      rating.
    category: Harm category.
    influentialTerms: The influential terms that could potentially block the
      response.
    probability: Harm probability levels in the content.
    probabilityScore: Harm probability score.
    severity: Harm severity levels in the content.
    severityScore: Harm severity score.
  """

    class CategoryValueValuesEnum(_messages.Enum):
        """Harm category.

    Values:
      HARM_CATEGORY_UNSPECIFIED: The harm category is unspecified.
      HARM_CATEGORY_HATE_SPEECH: The harm category is hate speech.
      HARM_CATEGORY_DANGEROUS_CONTENT: The harm category is dengerous content.
      HARM_CATEGORY_HARASSMENT: The harm category is harassment.
      HARM_CATEGORY_SEXUALLY_EXPLICIT: The harm category is sexually explicit.
    """
        HARM_CATEGORY_UNSPECIFIED = 0
        HARM_CATEGORY_HATE_SPEECH = 1
        HARM_CATEGORY_DANGEROUS_CONTENT = 2
        HARM_CATEGORY_HARASSMENT = 3
        HARM_CATEGORY_SEXUALLY_EXPLICIT = 4

    class ProbabilityValueValuesEnum(_messages.Enum):
        """Harm probability levels in the content.

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
        """Harm severity levels in the content.

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
    influentialTerms = _messages.MessageField('CloudAiNlLlmProtoServiceSafetyRatingInfluentialTerm', 3, repeated=True)
    probability = _messages.EnumField('ProbabilityValueValuesEnum', 4)
    probabilityScore = _messages.FloatField(5, variant=_messages.Variant.FLOAT)
    severity = _messages.EnumField('SeverityValueValuesEnum', 6)
    severityScore = _messages.FloatField(7, variant=_messages.Variant.FLOAT)