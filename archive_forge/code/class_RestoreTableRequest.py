from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RestoreTableRequest(_messages.Message):
    """The request for RestoreTable.

  Fields:
    backup: Name of the backup from which to restore. Values are of the form
      `projects//instances//clusters//backups/`.
    tableId: Required. The id of the table to create and restore to. This
      table must not already exist. The `table_id` appended to `parent` forms
      the full table name of the form `projects//instances//tables/`.
  """
    backup = _messages.StringField(1)
    tableId = _messages.StringField(2)