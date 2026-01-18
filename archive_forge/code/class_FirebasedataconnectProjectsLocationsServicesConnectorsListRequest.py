from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirebasedataconnectProjectsLocationsServicesConnectorsListRequest(_messages.Message):
    """A FirebasedataconnectProjectsLocationsServicesConnectorsListRequest
  object.

  Fields:
    filter: Optional. Filtering results.
    orderBy: Optional. Hint for how to order the results.
    pageSize: Optional. Requested page size. Server may return fewer items
      than requested. If unspecified, server will pick an appropriate default.
    pageToken: Optional. A page token, received from a previous
      `ListConnectors` call. Provide this to retrieve the subsequent page.
      When paginating, all other parameters provided to `ListConnectors` must
      match the call that provided the page token.
    parent: Required. Value of parent.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)