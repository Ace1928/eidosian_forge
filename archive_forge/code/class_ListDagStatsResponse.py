from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListDagStatsResponse(_messages.Message):
    """Response to `ListDagStatsRequest`.

  Fields:
    dagStats: List of DAGs with statistics.
    nextPageToken: The page token used to query for the next page if one
      exists.
  """
    dagStats = _messages.MessageField('DagStats', 1, repeated=True)
    nextPageToken = _messages.StringField(2)