from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmAttachmentDetails(_messages.Message):
    """Details for attachment of the disk to a VM.

  Fields:
    deviceName: Optional. Specifies a unique device name of your choice that
      is reflected into the /dev/disk/by-id/google-* tree of a Linux operating
      system running within the instance. If not specified, the server chooses
      a default device name to apply to this disk, in the form persistent-
      disk-x, where x is a number assigned by Google Compute Engine. This
      field is only applicable for persistent disks.
  """
    deviceName = _messages.StringField(1)