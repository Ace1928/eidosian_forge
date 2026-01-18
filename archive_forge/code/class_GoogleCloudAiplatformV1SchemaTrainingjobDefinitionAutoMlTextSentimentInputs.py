from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaTrainingjobDefinitionAutoMlTextSentimentInputs(_messages.Message):
    """A
  GoogleCloudAiplatformV1SchemaTrainingjobDefinitionAutoMlTextSentimentInputs
  object.

  Fields:
    sentimentMax: A sentiment is expressed as an integer ordinal, where higher
      value means a more positive sentiment. The range of sentiments that will
      be used is between 0 and sentimentMax (inclusive on both ends), and all
      the values in the range must be represented in the dataset before a
      model can be created. Only the Annotations with this sentimentMax will
      be used for training. sentimentMax value must be between 1 and 10
      (inclusive).
  """
    sentimentMax = _messages.IntegerField(1, variant=_messages.Variant.INT32)