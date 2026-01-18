from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1StreamingPredictRequest(_messages.Message):
    """Request message for PredictionService.StreamingPredict. The first
  message must contain endpoint field and optionally input. The subsequent
  messages must contain input.

  Fields:
    inputs: The prediction input.
    parameters: The parameters that govern the prediction.
  """
    inputs = _messages.MessageField('GoogleCloudAiplatformV1beta1Tensor', 1, repeated=True)
    parameters = _messages.MessageField('GoogleCloudAiplatformV1beta1Tensor', 2)