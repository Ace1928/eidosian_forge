from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ListFindingTypeStatsResponse(_messages.Message):
    """Response for the `ListFindingTypeStats` method.

  Fields:
    findingTypeStats: The list of FindingTypeStats returned.
  """
    findingTypeStats = _messages.MessageField('FindingTypeStats', 1, repeated=True)