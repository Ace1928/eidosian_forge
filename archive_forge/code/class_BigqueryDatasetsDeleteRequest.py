from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigqueryDatasetsDeleteRequest(_messages.Message):
    """A BigqueryDatasetsDeleteRequest object.

  Fields:
    datasetId: Dataset ID of dataset being deleted
    deleteContents: If True, delete all the tables in the dataset. If False
      and the dataset contains tables, the request will fail. Default is False
    projectId: Project ID of the dataset being deleted
  """
    datasetId = _messages.StringField(1, required=True)
    deleteContents = _messages.BooleanField(2)
    projectId = _messages.StringField(3, required=True)