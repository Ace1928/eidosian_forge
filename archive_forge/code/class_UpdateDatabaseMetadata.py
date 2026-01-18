from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpdateDatabaseMetadata(_messages.Message):
    """Metadata type for the operation returned by UpdateDatabase.

  Fields:
    cancelTime: The time at which this operation was cancelled. If set, this
      operation is in the process of undoing itself (which is best-effort).
    progress: The progress of the UpdateDatabase operation.
    request: The request for UpdateDatabase.
  """
    cancelTime = _messages.StringField(1)
    progress = _messages.MessageField('OperationProgress', 2)
    request = _messages.MessageField('UpdateDatabaseRequest', 3)