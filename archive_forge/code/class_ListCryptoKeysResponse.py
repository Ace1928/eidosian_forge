from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListCryptoKeysResponse(_messages.Message):
    """Response message for KeyManagementService.ListCryptoKeys.

  Fields:
    cryptoKeys: The list of CryptoKeys.
    nextPageToken: A token to retrieve next page of results. Pass this value
      in ListCryptoKeysRequest.page_token to retrieve the next page of
      results.
    totalSize: The total number of CryptoKeys that matched the query.
  """
    cryptoKeys = _messages.MessageField('CryptoKey', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    totalSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)