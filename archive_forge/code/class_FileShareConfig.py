from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FileShareConfig(_messages.Message):
    """File share configuration for the instance.

  Fields:
    capacityGb: File share capacity in gigabytes (GB). Filestore defines 1 GB
      as 1024^3 bytes.
    name: Required. The name of the file share. Must use 1-16 characters for
      the basic service tier and 1-63 characters for all other service tiers.
      Must use lowercase letters, numbers, or underscores `[a-z0-9_]`. Must
      start with a letter. Immutable.
    nfsExportOptions: Nfs Export Options. There is a limit of 10 export
      options per file share.
    sourceBackup: The resource name of the backup, in the format
      `projects/{project_number}/locations/{location_id}/backups/{backup_id}`,
      that this file share has been restored from.
  """
    capacityGb = _messages.IntegerField(1)
    name = _messages.StringField(2)
    nfsExportOptions = _messages.MessageField('NfsExportOptions', 3, repeated=True)
    sourceBackup = _messages.StringField(4)