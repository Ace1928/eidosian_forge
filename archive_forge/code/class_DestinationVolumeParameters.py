from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DestinationVolumeParameters(_messages.Message):
    """DestinationVolumeParameters specify input parameters used for creating
  destination volume.

  Fields:
    description: Description for the destination volume.
    shareName: Destination volume's share name. If not specified, source
      volume's share name will be used.
    storagePool: Required. Existing destination StoragePool name.
    volumeId: Desired destination volume resource id. If not specified, source
      volume's resource id will be used. This value must start with a
      lowercase letter followed by up to 62 lowercase letters, numbers, or
      hyphens, and cannot end with a hyphen.
  """
    description = _messages.StringField(1)
    shareName = _messages.StringField(2)
    storagePool = _messages.StringField(3)
    volumeId = _messages.StringField(4)