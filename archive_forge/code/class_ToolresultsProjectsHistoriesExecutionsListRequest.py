from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ToolresultsProjectsHistoriesExecutionsListRequest(_messages.Message):
    """A ToolresultsProjectsHistoriesExecutionsListRequest object.

  Fields:
    historyId: A History id. Required.
    pageSize: The maximum number of Executions to fetch. Default value: 25.
      The server will use this default if the field is not set or has a value
      of 0. Optional.
    pageToken: A continuation token to resume the query at the next item.
      Optional.
    projectId: A Project id. Required.
  """
    historyId = _messages.StringField(1, required=True)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    projectId = _messages.StringField(4, required=True)