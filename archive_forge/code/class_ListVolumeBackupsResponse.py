from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListVolumeBackupsResponse(_messages.Message):
    """Response message for ListVolumeBackups.

  Fields:
    nextPageToken: A token which may be sent as page_token in a subsequent
      `ListVolumeBackups` call to retrieve the next page of results. If this
      field is omitted or empty, then there are no more results to return.
    volumeBackups: The list of VolumeBackups matching the given criteria.
  """
    nextPageToken = _messages.StringField(1)
    volumeBackups = _messages.MessageField('VolumeBackup', 2, repeated=True)