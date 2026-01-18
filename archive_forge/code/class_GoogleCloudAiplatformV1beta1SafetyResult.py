from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SafetyResult(_messages.Message):
    """Spec for safety result.

  Fields:
    confidence: Output only. Confidence for safety score.
    explanation: Output only. Explanation for safety score.
    score: Output only. Safety score.
  """
    confidence = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    explanation = _messages.StringField(2)
    score = _messages.FloatField(3, variant=_messages.Variant.FLOAT)