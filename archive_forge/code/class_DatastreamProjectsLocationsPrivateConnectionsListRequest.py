from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatastreamProjectsLocationsPrivateConnectionsListRequest(_messages.Message):
    """A DatastreamProjectsLocationsPrivateConnectionsListRequest object.

  Fields:
    filter: Filter request.
    orderBy: Order by fields for the result.
    pageSize: Maximum number of private connectivity configurations to return.
      If unspecified, at most 50 private connectivity configurations that will
      be returned. The maximum value is 1000; values above 1000 will be
      coerced to 1000.
    pageToken: Page token received from a previous `ListPrivateConnections`
      call. Provide this to retrieve the subsequent page. When paginating, all
      other parameters provided to `ListPrivateConnections` must match the
      call that provided the page token.
    parent: Required. The parent that owns the collection of private
      connectivity configurations.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)