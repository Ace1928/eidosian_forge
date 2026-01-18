from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DiskConfig(_messages.Message):
    """Specifies the config of disk options for a group of VM instances.

  Fields:
    bootDiskSizeGb: Optional. Size in GB of the boot disk (default is 500GB).
    bootDiskType: Optional. Type of the boot disk (default is "pd-standard").
      Valid values: "pd-balanced" (Persistent Disk Balanced Solid State
      Drive), "pd-ssd" (Persistent Disk Solid State Drive), or "pd-standard"
      (Persistent Disk Hard Disk Drive). See Disk types
      (https://cloud.google.com/compute/docs/disks#disk-types).
    localSsdInterface: Optional. Interface type of local SSDs (default is
      "scsi"). Valid values: "scsi" (Small Computer System Interface), "nvme"
      (Non-Volatile Memory Express). See local SSD performance
      (https://cloud.google.com/compute/docs/disks/local-ssd#performance).
    numLocalSsds: Optional. Number of attached SSDs, from 0 to 8 (default is
      0). If SSDs are not attached, the boot disk is used to store runtime
      logs and HDFS
      (https://hadoop.apache.org/docs/r1.2.1/hdfs_user_guide.html) data. If
      one or more SSDs are attached, this runtime bulk data is spread across
      them, and the boot disk contains only basic config and installed
      binaries.Note: Local SSD options may vary by machine type and number of
      vCPUs selected.
  """
    bootDiskSizeGb = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    bootDiskType = _messages.StringField(2)
    localSsdInterface = _messages.StringField(3)
    numLocalSsds = _messages.IntegerField(4, variant=_messages.Variant.INT32)