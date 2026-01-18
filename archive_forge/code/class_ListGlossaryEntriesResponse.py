from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListGlossaryEntriesResponse(_messages.Message):
    """Response message for ListGlossaryEntries

  Fields:
    glossaryEntries: Optional. The Glossary Entries
    nextPageToken: Optional. A token to retrieve a page of results. Pass this
      value in the [ListGLossaryEntriesRequest.page_token] field in the
      subsequent calls.
  """
    glossaryEntries = _messages.MessageField('GlossaryEntry', 1, repeated=True)
    nextPageToken = _messages.StringField(2)