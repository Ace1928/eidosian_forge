from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DmlStatistics(_messages.Message):
    """Detailed statistics for DML statements

  Fields:
    deletedRowCount: Output only. Number of deleted Rows. populated by DML
      DELETE, MERGE and TRUNCATE statements.
    insertedRowCount: Output only. Number of inserted Rows. Populated by DML
      INSERT and MERGE statements
    updatedRowCount: Output only. Number of updated Rows. Populated by DML
      UPDATE and MERGE statements.
  """
    deletedRowCount = _messages.IntegerField(1)
    insertedRowCount = _messages.IntegerField(2)
    updatedRowCount = _messages.IntegerField(3)