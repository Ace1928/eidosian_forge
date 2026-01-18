from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1GenerationConfig(_messages.Message):
    """Generation config.

  Fields:
    candidateCount: Optional. Number of candidates to generate.
    maxOutputTokens: Optional. The maximum number of output tokens to generate
      per message.
    stopSequences: Optional. Stop sequences.
    temperature: Optional. Controls the randomness of predictions.
    topK: Optional. If specified, top-k sampling will be used.
    topP: Optional. If specified, nucleus sampling will be used.
  """
    candidateCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    maxOutputTokens = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    stopSequences = _messages.StringField(3, repeated=True)
    temperature = _messages.FloatField(4, variant=_messages.Variant.FLOAT)
    topK = _messages.FloatField(5, variant=_messages.Variant.FLOAT)
    topP = _messages.FloatField(6, variant=_messages.Variant.FLOAT)