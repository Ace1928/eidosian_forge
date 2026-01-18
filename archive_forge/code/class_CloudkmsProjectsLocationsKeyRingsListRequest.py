from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudkmsProjectsLocationsKeyRingsListRequest(_messages.Message):
    """A CloudkmsProjectsLocationsKeyRingsListRequest object.

  Fields:
    pageSize: Optional limit on the number of KeyRings to include in the
      response.  Further KeyRings can subsequently be obtained by including
      the ListKeyRingsResponse.next_page_token in a subsequent request.  If
      unspecified, the server will pick an appropriate default.
    pageToken: Optional pagination token, returned earlier via
      ListKeyRingsResponse.next_page_token.
    parent: Required. The resource name of the location associated with the
      KeyRings, in the format `projects/*/locations/*`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)