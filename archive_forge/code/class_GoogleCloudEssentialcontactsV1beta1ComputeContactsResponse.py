from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GoogleCloudEssentialcontactsV1beta1ComputeContactsResponse(_messages.Message):
    """Response message for the ComputeContacts method.

  Fields:
    contacts: All contacts for the resource that are subscribed to the
      specified notification categories, including contacts inherited from any
      parent resources.
    nextPageToken: If there are more results than those appearing in this
      response, then `next_page_token` is included. To get the next set of
      results, call this method again using the value of `next_page_token` as
      `page_token` and the rest of the parameters the same as the original
      request.
  """
    contacts = _messages.MessageField('GoogleCloudEssentialcontactsV1beta1Contact', 1, repeated=True)
    nextPageToken = _messages.StringField(2)