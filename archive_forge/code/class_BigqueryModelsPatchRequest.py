from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigqueryModelsPatchRequest(_messages.Message):
    """A BigqueryModelsPatchRequest object.

  Fields:
    datasetId: Required. Dataset ID of the model to patch.
    model: A Model resource to be passed as the request body.
    modelId: Required. Model ID of the model to patch.
    projectId: Required. Project ID of the model to patch.
  """
    datasetId = _messages.StringField(1, required=True)
    model = _messages.MessageField('Model', 2)
    modelId = _messages.StringField(3, required=True)
    projectId = _messages.StringField(4, required=True)