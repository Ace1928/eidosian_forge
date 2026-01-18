from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListReferencesResponse(_messages.Message):
    """The ListReferencesResponse response.

  Fields:
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
    references: The list of references.
  """
    nextPageToken = _messages.StringField(1)
    references = _messages.MessageField('Reference', 2, repeated=True)