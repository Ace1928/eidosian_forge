from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpdateDatabaseDdlMetadata(_messages.Message):
    """Metadata type for the operation returned by UpdateDatabaseDdl.

  Fields:
    actions: The brief action info for the DDL statements. `actions[i]` is the
      brief info for `statements[i]`.
    commitTimestamps: Reports the commit timestamps of all statements that
      have succeeded so far, where `commit_timestamps[i]` is the commit
      timestamp for the statement `statements[i]`.
    database: The database being modified.
    progress: The progress of the UpdateDatabaseDdl operations. All DDL
      statements will have continuously updating progress, and `progress[i]`
      is the operation progress for `statements[i]`. Also, `progress[i]` will
      have start time and end time populated with commit timestamp of
      operation, as well as a progress of 100% once the operation has
      completed.
    statements: For an update this list contains all the statements. For an
      individual statement, this list contains only that statement.
    throttled: Output only. When true, indicates that the operation is
      throttled e.g. due to resource constraints. When resources become
      available the operation will resume and this field will be false again.
  """
    actions = _messages.MessageField('DdlStatementActionInfo', 1, repeated=True)
    commitTimestamps = _messages.StringField(2, repeated=True)
    database = _messages.StringField(3)
    progress = _messages.MessageField('OperationProgress', 4, repeated=True)
    statements = _messages.StringField(5, repeated=True)
    throttled = _messages.BooleanField(6)