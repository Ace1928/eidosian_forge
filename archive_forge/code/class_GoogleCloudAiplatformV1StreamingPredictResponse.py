from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1StreamingPredictResponse(_messages.Message):
    """Response message for PredictionService.StreamingPredict.

  Fields:
    outputs: The prediction output.
    parameters: The parameters that govern the prediction.
  """
    outputs = _messages.MessageField('GoogleCloudAiplatformV1Tensor', 1, repeated=True)
    parameters = _messages.MessageField('GoogleCloudAiplatformV1Tensor', 2)