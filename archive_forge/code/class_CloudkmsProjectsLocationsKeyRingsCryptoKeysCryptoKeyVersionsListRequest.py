from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsListRequest(_messages.Message):
    """A CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsListRequest
  object.

  Fields:
    pageSize: Optional limit on the number of CryptoKeyVersions to include in
      the response. Further CryptoKeyVersions can subsequently be obtained by
      including the ListCryptoKeyVersionsResponse.next_page_token in a
      subsequent request. If unspecified, the server will pick an appropriate
      default.
    pageToken: Optional pagination token, returned earlier via
      ListCryptoKeyVersionsResponse.next_page_token.
    parent: Required. The resource name of the CryptoKey to list, in the
      format `projects/*/locations/*/keyRings/*/cryptoKeys/*`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)