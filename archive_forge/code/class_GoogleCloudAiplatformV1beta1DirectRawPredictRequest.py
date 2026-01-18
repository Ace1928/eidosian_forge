from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1DirectRawPredictRequest(_messages.Message):
    """Request message for PredictionService.DirectRawPredict.

  Fields:
    input: The prediction input.
    methodName: Fully qualified name of the API method being invoked to
      perform predictions. Format: `/namespace.Service/Method/` Example:
      `/tensorflow.serving.PredictionService/Predict`
  """
    input = _messages.BytesField(1)
    methodName = _messages.StringField(2)