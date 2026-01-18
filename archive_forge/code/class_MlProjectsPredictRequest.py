from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MlProjectsPredictRequest(_messages.Message):
    """A MlProjectsPredictRequest object.

  Fields:
    googleCloudMlV1PredictRequest: A GoogleCloudMlV1PredictRequest resource to
      be passed as the request body.
    name: Required. The resource name of a model or a version. Authorization:
      requires the `predict` permission on the specified resource.
  """
    googleCloudMlV1PredictRequest = _messages.MessageField('GoogleCloudMlV1PredictRequest', 1)
    name = _messages.StringField(2, required=True)