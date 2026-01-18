from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class BackupRunsListResponse(_messages.Message):
    """Backup run list results.

  Fields:
    items: A list of backup runs in reverse chronological order of the
      enqueued time.
    kind: This is always `sql#backupRunsList`.
    nextPageToken: The continuation token, used to page through large result
      sets. Provide this value in a subsequent request to return the next page
      of results.
  """
    items = _messages.MessageField('BackupRun', 1, repeated=True)
    kind = _messages.StringField(2)
    nextPageToken = _messages.StringField(3)