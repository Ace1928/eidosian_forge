from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListBackupOperationsResponse(_messages.Message):
    """The response for ListBackupOperations.

  Fields:
    nextPageToken: `next_page_token` can be sent in a subsequent
      ListBackupOperations call to fetch more of the matching metadata.
    operations: The list of matching backup long-running operations. Each
      operation's name will be prefixed by the backup's name. The operation's
      metadata field type `metadata.type_url` describes the type of the
      metadata. Operations returned include those that are pending or have
      completed/failed/canceled within the last 7 days. Operations returned
      are ordered by `operation.metadata.value.progress.start_time` in
      descending order starting from the most recently started operation.
  """
    nextPageToken = _messages.StringField(1)
    operations = _messages.MessageField('Operation', 2, repeated=True)