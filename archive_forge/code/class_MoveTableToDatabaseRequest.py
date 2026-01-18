from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MoveTableToDatabaseRequest(_messages.Message):
    """Request message for DataprocMetastore.MoveTableToDatabase.

  Fields:
    dbName: Required. The name of the database where the table resides.
    destinationDbName: Required. The name of the database where the table
      should be moved.
    tableName: Required. The name of the table to be moved.
  """
    dbName = _messages.StringField(1)
    destinationDbName = _messages.StringField(2)
    tableName = _messages.StringField(3)