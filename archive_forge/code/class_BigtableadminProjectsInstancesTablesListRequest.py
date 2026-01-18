from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigtableadminProjectsInstancesTablesListRequest(_messages.Message):
    """A BigtableadminProjectsInstancesTablesListRequest object.

  Enums:
    ViewValueValuesEnum: The view to be applied to the returned tables'
      fields. Only NAME_ONLY view (default), REPLICATION_VIEW and
      ENCRYPTION_VIEW are supported.

  Fields:
    pageSize: Maximum number of results per page. A page_size of zero lets the
      server choose the number of items to return. A page_size which is
      strictly positive will return at most that many items. A negative
      page_size will cause an error. Following the first request, subsequent
      paginated calls are not required to pass a page_size. If a page_size is
      set in subsequent calls, it must match the page_size given in the first
      request.
    pageToken: The value of `next_page_token` returned by a previous call.
    parent: Required. The unique name of the instance for which tables should
      be listed. Values are of the form
      `projects/{project}/instances/{instance}`.
    view: The view to be applied to the returned tables' fields. Only
      NAME_ONLY view (default), REPLICATION_VIEW and ENCRYPTION_VIEW are
      supported.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """The view to be applied to the returned tables' fields. Only NAME_ONLY
    view (default), REPLICATION_VIEW and ENCRYPTION_VIEW are supported.

    Values:
      VIEW_UNSPECIFIED: Uses the default view for each method as documented in
        its request.
      NAME_ONLY: Only populates `name`.
      SCHEMA_VIEW: Only populates `name` and fields related to the table's
        schema.
      REPLICATION_VIEW: Only populates `name` and fields related to the
        table's replication state.
      ENCRYPTION_VIEW: Only populates `name` and fields related to the table's
        encryption state.
      STATS_VIEW: Only populates `name` and fields related to the table's
        stats (e.g. TableStats and ColumnFamilyStats).
      FULL: Populates all fields except for stats. See STATS_VIEW to request
        stats.
    """
        VIEW_UNSPECIFIED = 0
        NAME_ONLY = 1
        SCHEMA_VIEW = 2
        REPLICATION_VIEW = 3
        ENCRYPTION_VIEW = 4
        STATS_VIEW = 5
        FULL = 6
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 4)