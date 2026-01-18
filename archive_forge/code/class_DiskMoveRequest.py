from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DiskMoveRequest(_messages.Message):
    """A DiskMoveRequest object.

  Fields:
    destinationZone: The URL of the destination zone to move the disk. This
      can be a full or partial URL. For example, the following are all valid
      URLs to a zone: -
      https://www.googleapis.com/compute/v1/projects/project/zones/zone -
      projects/project/zones/zone - zones/zone
    targetDisk: The URL of the target disk to move. This can be a full or
      partial URL. For example, the following are all valid URLs to a disk: -
      https://www.googleapis.com/compute/v1/projects/project/zones/zone
      /disks/disk - projects/project/zones/zone/disks/disk -
      zones/zone/disks/disk
  """
    destinationZone = _messages.StringField(1)
    targetDisk = _messages.StringField(2)