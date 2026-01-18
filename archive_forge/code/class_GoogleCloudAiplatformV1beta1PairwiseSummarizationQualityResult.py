from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1PairwiseSummarizationQualityResult(_messages.Message):
    """Spec for pairwise summarization quality result.

  Enums:
    PairwiseChoiceValueValuesEnum: Output only. Pairwise summarization
      prediction choice.

  Fields:
    confidence: Output only. Confidence for summarization quality score.
    explanation: Output only. Explanation for summarization quality score.
    pairwiseChoice: Output only. Pairwise summarization prediction choice.
  """

    class PairwiseChoiceValueValuesEnum(_messages.Enum):
        """Output only. Pairwise summarization prediction choice.

    Values:
      PAIRWISE_CHOICE_UNSPECIFIED: Unspecified prediction choice.
      BASELINE: Baseline prediction wins
      CANDIDATE: Candidate prediction wins
      TIE: Winner cannot be determined
    """
        PAIRWISE_CHOICE_UNSPECIFIED = 0
        BASELINE = 1
        CANDIDATE = 2
        TIE = 3
    confidence = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    explanation = _messages.StringField(2)
    pairwiseChoice = _messages.EnumField('PairwiseChoiceValueValuesEnum', 3)