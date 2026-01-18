from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GoogleCloudEssentialcontactsV1ListContactsResponse(_messages.Message):
    """Response message for the ListContacts method.

  Fields:
    contacts: The contacts for the specified resource.
    nextPageToken: If there are more results than those appearing in this
      response, then `next_page_token` is included. To get the next set of
      results, call this method again using the value of `next_page_token` as
      `page_token` and the rest of the parameters the same as the original
      request.
  """
    contacts = _messages.MessageField('GoogleCloudEssentialcontactsV1Contact', 1, repeated=True)
    nextPageToken = _messages.StringField(2)