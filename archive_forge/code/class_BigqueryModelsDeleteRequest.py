from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigqueryModelsDeleteRequest(_messages.Message):
    """A BigqueryModelsDeleteRequest object.

  Fields:
    datasetId: Required. Dataset ID of the model to delete.
    modelId: Required. Model ID of the model to delete.
    projectId: Required. Project ID of the model to delete.
  """
    datasetId = _messages.StringField(1, required=True)
    modelId = _messages.StringField(2, required=True)
    projectId = _messages.StringField(3, required=True)