from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1PredictRequest(_messages.Message):
    """Request for predictions to be issued against a trained model.

  Fields:
    httpBody:  Required. The prediction request body. Refer to the [request
      body details section](#request-body-details) for more information on how
      to structure your request.
  """
    httpBody = _messages.MessageField('GoogleApiHttpBody', 1)