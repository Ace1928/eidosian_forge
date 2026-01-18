from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListViewsResponse(_messages.Message):
    """The response from ListViews.

  Fields:
    nextPageToken: If there might be more results than appear in this
      response, then nextPageToken is included. To get the next set of
      results, call the same method again using the value of nextPageToken as
      pageToken.
    views: A list of views.
  """
    nextPageToken = _messages.StringField(1)
    views = _messages.MessageField('LogView', 2, repeated=True)