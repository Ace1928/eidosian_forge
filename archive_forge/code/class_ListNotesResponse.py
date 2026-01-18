from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListNotesResponse(_messages.Message):
    """Response for listing notes.

  Fields:
    nextPageToken: The next pagination token in the list response. It should
      be used as `page_token` for the following request. An empty value means
      no more results.
    notes: The notes requested.
  """
    nextPageToken = _messages.StringField(1)
    notes = _messages.MessageField('Note', 2, repeated=True)