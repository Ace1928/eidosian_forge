from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CustomerEncryptionKeyProtectedDisk(_messages.Message):
    """A CustomerEncryptionKeyProtectedDisk object.

  Fields:
    diskEncryptionKey: Decrypts data associated with the disk with a customer-
      supplied encryption key.
    source: Specifies a valid partial or full URL to an existing Persistent
      Disk resource. This field is only applicable for persistent disks. For
      example: "source": "/compute/v1/projects/project_id/zones/zone/disks/
      disk_name
  """
    diskEncryptionKey = _messages.MessageField('CustomerEncryptionKey', 1)
    source = _messages.StringField(2)