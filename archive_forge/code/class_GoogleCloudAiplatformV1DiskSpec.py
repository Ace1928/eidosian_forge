from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1DiskSpec(_messages.Message):
    """Represents the spec of disk options.

  Fields:
    bootDiskSizeGb: Size in GB of the boot disk (default is 100GB).
    bootDiskType: Type of the boot disk (default is "pd-ssd"). Valid values:
      "pd-ssd" (Persistent Disk Solid State Drive) or "pd-standard"
      (Persistent Disk Hard Disk Drive).
  """
    bootDiskSizeGb = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    bootDiskType = _messages.StringField(2)