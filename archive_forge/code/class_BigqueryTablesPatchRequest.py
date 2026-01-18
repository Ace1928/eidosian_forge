from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigqueryTablesPatchRequest(_messages.Message):
    """A BigqueryTablesPatchRequest object.

  Fields:
    datasetId: Dataset ID of the table to update
    projectId: Project ID of the table to update
    table: A Table resource to be passed as the request body.
    tableId: Table ID of the table to update
  """
    datasetId = _messages.StringField(1, required=True)
    projectId = _messages.StringField(2, required=True)
    table = _messages.MessageField('Table', 3)
    tableId = _messages.StringField(4, required=True)