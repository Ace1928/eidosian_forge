from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListSavedQueriesResponse(_messages.Message):
    """Response of listing saved queries.

  Fields:
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    savedQueries: A list of savedQueries.
  """
    nextPageToken = _messages.StringField(1)
    savedQueries = _messages.MessageField('SavedQuery', 2, repeated=True)