from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListHistoriesResponse(_messages.Message):
    """Response message for HistoryService.List

  Fields:
    histories: Histories.
    nextPageToken: A continuation token to resume the query at the next item.
      Will only be set if there are more histories to fetch. Tokens are valid
      for up to one hour from the time of the first list request. For
      instance, if you make a list request at 1PM and use the token from this
      first request 10 minutes later, the token from this second response will
      only be valid for 50 minutes.
  """
    histories = _messages.MessageField('History', 1, repeated=True)
    nextPageToken = _messages.StringField(2)