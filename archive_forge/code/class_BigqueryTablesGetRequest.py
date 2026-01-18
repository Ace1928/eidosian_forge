from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigqueryTablesGetRequest(_messages.Message):
    """A BigqueryTablesGetRequest object.

  Fields:
    datasetId: Dataset ID of the requested table
    projectId: Project ID of the requested table
    tableId: Table ID of the requested table
  """
    datasetId = _messages.StringField(1, required=True)
    projectId = _messages.StringField(2, required=True)
    tableId = _messages.StringField(3, required=True)