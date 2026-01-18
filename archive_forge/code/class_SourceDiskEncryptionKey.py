from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SourceDiskEncryptionKey(_messages.Message):
    """A SourceDiskEncryptionKey object.

  Fields:
    diskEncryptionKey: The customer-supplied encryption key of the source
      disk. Required if the source disk is protected by a customer-supplied
      encryption key.
    sourceDisk: URL of the disk attached to the source instance. This can be a
      full or valid partial URL. For example, the following are valid values:
      - https://www.googleapis.com/compute/v1/projects/project/zones/zone
      /disks/disk - projects/project/zones/zone/disks/disk -
      zones/zone/disks/disk
  """
    diskEncryptionKey = _messages.MessageField('CustomerEncryptionKey', 1)
    sourceDisk = _messages.StringField(2)