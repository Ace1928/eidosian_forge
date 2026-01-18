from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListGlossariesResponse(_messages.Message):
    """Response message for ListGlossaries.

  Fields:
    glossaries: The list of glossaries for a project.
    nextPageToken: A token to retrieve a page of results. Pass this value in
      the [ListGlossariesRequest.page_token] field in the subsequent call to
      `ListGlossaries` method to retrieve the next page of results.
  """
    glossaries = _messages.MessageField('Glossary', 1, repeated=True)
    nextPageToken = _messages.StringField(2)