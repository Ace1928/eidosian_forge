from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1StreamRawPredictRequest(_messages.Message):
    """Request message for PredictionService.StreamRawPredict.

  Fields:
    httpBody: The prediction input. Supports HTTP headers and arbitrary data
      payload.
  """
    httpBody = _messages.MessageField('GoogleApiHttpBody', 1)