from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListDagRunsResponse(_messages.Message):
    """Response to `ListDagRunsRequest`.

  Fields:
    dagRuns: The list of DAG runs returned.
    nextPageToken: The page token used to query for the next page if one
      exists.
  """
    dagRuns = _messages.MessageField('DagRun', 1, repeated=True)
    nextPageToken = _messages.StringField(2)