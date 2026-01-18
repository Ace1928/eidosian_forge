from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListLiensResponse(_messages.Message):
    """The response message for Liens.ListLiens.

  Fields:
    liens: A list of Liens.
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
  """
    liens = _messages.MessageField('Lien', 1, repeated=True)
    nextPageToken = _messages.StringField(2)