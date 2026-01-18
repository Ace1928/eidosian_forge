from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SummarizationQualityResult(_messages.Message):
    """Spec for summarization quality result.

  Fields:
    confidence: Output only. Confidence for summarization quality score.
    explanation: Output only. Explanation for summarization quality score.
    score: Output only. Summarization Quality score.
  """
    confidence = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    explanation = _messages.StringField(2)
    score = _messages.FloatField(3, variant=_messages.Variant.FLOAT)