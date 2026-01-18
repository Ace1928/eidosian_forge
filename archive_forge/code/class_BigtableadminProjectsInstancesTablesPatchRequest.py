from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigtableadminProjectsInstancesTablesPatchRequest(_messages.Message):
    """A BigtableadminProjectsInstancesTablesPatchRequest object.

  Fields:
    name: The unique name of the table. Values are of the form
      `projects/{project}/instances/{instance}/tables/_a-zA-Z0-9*`. Views:
      `NAME_ONLY`, `SCHEMA_VIEW`, `REPLICATION_VIEW`, `STATS_VIEW`, `FULL`
    table: A Table resource to be passed as the request body.
    updateMask: Required. The list of fields to update. A mask specifying
      which fields (e.g. `change_stream_config`) in the `table` field should
      be updated. This mask is relative to the `table` field, not to the
      request message. The wildcard (*) path is currently not supported.
      Currently UpdateTable is only supported for the following fields: *
      `change_stream_config` * `change_stream_config.retention_period` *
      `deletion_protection` If `column_families` is set in `update_mask`, it
      will return an UNIMPLEMENTED error.
  """
    name = _messages.StringField(1, required=True)
    table = _messages.MessageField('Table', 2)
    updateMask = _messages.StringField(3)