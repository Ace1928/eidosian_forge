from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatamigrationProjectsLocationsPrivateConnectionsListRequest(_messages.Message):
    """A DatamigrationProjectsLocationsPrivateConnectionsListRequest object.

  Fields:
    filter: A filter expression that filters private connections listed in the
      response. The expression must specify the field name, a comparison
      operator, and the value that you want to use for filtering. The value
      must be a string, a number, or a boolean. The comparison operator must
      be either =, !=, >, or <. For example, list private connections created
      this year by specifying **createTime %gt;
      2021-01-01T00:00:00.000000000Z**.
    orderBy: Order by fields for the result.
    pageSize: Maximum number of private connections to return. If unspecified,
      at most 50 private connections that are returned. The maximum value is
      1000; values above 1000 are coerced to 1000.
    pageToken: Page token received from a previous `ListPrivateConnections`
      call. Provide this to retrieve the subsequent page. When paginating, all
      other parameters provided to `ListPrivateConnections` must match the
      call that provided the page token.
    parent: Required. The parent that owns the collection of private
      connections.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)