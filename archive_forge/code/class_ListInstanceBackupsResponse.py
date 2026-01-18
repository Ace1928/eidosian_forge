from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListInstanceBackupsResponse(_messages.Message):
    """Response from listing Looker instance backups.

  Fields:
    instanceBackups: The list of instances matching the request filters, up to
      the requested `page_size`.
    nextPageToken: If provided, a page token that can look up the next
      `page_size` results. If empty, the results list is exhausted.
    unreachable: Locations that could not be reached.
  """
    instanceBackups = _messages.MessageField('InstanceBackup', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachable = _messages.StringField(3, repeated=True)