from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RegionDisksStartAsyncReplicationRequest(_messages.Message):
    """A RegionDisksStartAsyncReplicationRequest object.

  Fields:
    asyncSecondaryDisk: The secondary disk to start asynchronous replication
      to. You can provide this as a partial or full URL to the resource. For
      example, the following are valid values: -
      https://www.googleapis.com/compute/v1/projects/project/zones/zone
      /disks/disk -
      https://www.googleapis.com/compute/v1/projects/project/regions/region
      /disks/disk - projects/project/zones/zone/disks/disk -
      projects/project/regions/region/disks/disk - zones/zone/disks/disk -
      regions/region/disks/disk
  """
    asyncSecondaryDisk = _messages.StringField(1)