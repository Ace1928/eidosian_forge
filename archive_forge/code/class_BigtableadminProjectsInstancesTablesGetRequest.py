from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigtableadminProjectsInstancesTablesGetRequest(_messages.Message):
    """A BigtableadminProjectsInstancesTablesGetRequest object.

  Enums:
    ViewValueValuesEnum: The view to be applied to the returned table's
      fields. Defaults to `SCHEMA_VIEW` if unspecified.

  Fields:
    name: Required. The unique name of the requested table. Values are of the
      form `projects/{project}/instances/{instance}/tables/{table}`.
    view: The view to be applied to the returned table's fields. Defaults to
      `SCHEMA_VIEW` if unspecified.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """The view to be applied to the returned table's fields. Defaults to
    `SCHEMA_VIEW` if unspecified.

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
    name = _messages.StringField(1, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 2)