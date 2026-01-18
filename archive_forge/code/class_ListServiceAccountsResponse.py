from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ListServiceAccountsResponse(_messages.Message):
    """The service account list response.

  Fields:
    accounts: The list of matching service accounts.
    nextPageToken: To retrieve the next page of results, set
      ListServiceAccountsRequest.page_token to this value.
  """
    accounts = _messages.MessageField('ServiceAccount', 1, repeated=True)
    nextPageToken = _messages.StringField(2)