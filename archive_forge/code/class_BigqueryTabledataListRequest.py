from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigqueryTabledataListRequest(_messages.Message):
    """A BigqueryTabledataListRequest object.

  Fields:
    datasetId: Dataset ID of the table to read
    maxResults: Maximum number of results to return
    pageToken: Page token, returned by a previous call, identifying the result
      set
    projectId: Project ID of the table to read
    startIndex: Zero-based index of the starting row to read
    tableId: Table ID of the table to read
  """
    datasetId = _messages.StringField(1, required=True)
    maxResults = _messages.IntegerField(2, variant=_messages.Variant.UINT32)
    pageToken = _messages.StringField(3)
    projectId = _messages.StringField(4, required=True)
    startIndex = _messages.IntegerField(5, variant=_messages.Variant.UINT64)
    tableId = _messages.StringField(6, required=True)