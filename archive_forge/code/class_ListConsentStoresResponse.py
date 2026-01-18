from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListConsentStoresResponse(_messages.Message):
    """A ListConsentStoresResponse object.

  Fields:
    consentStores: The returned consent stores. The maximum number of stores
      returned is determined by the value of page_size in the
      ListConsentStoresRequest.
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
  """
    consentStores = _messages.MessageField('ConsentStore', 1, repeated=True)
    nextPageToken = _messages.StringField(2)