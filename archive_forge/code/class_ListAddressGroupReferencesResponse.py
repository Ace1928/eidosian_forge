from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListAddressGroupReferencesResponse(_messages.Message):
    """Response of the ListAddressGroupReferences method.

  Fields:
    addressGroupReferences: A list of references that matches the specified
      filter in the request.
    nextPageToken: If there might be more results than those appearing in this
      response, then `next_page_token` is included. To get the next set of
      results, call this method again using the value of `next_page_token` as
      `page_token`.
  """
    addressGroupReferences = _messages.MessageField('ListAddressGroupReferencesResponseAddressGroupReference', 1, repeated=True)
    nextPageToken = _messages.StringField(2)