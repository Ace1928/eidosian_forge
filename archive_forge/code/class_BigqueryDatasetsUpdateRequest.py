from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigqueryDatasetsUpdateRequest(_messages.Message):
    """A BigqueryDatasetsUpdateRequest object.

  Fields:
    dataset: A Dataset resource to be passed as the request body.
    datasetId: Dataset ID of the dataset being updated
    projectId: Project ID of the dataset being updated
  """
    dataset = _messages.MessageField('Dataset', 1)
    datasetId = _messages.StringField(2, required=True)
    projectId = _messages.StringField(3, required=True)