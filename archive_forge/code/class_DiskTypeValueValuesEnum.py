from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DiskTypeValueValuesEnum(_messages.Enum):
    """Output only. The disk type for this volume.

    Values:
      DISK_TYPE_UNSPECIFIED: This disk type for this volume is Unspecified.
      STANDARD: This disk type for this volume is Standard.
      SSD: The disk type for this volume is SSD.
    """
    DISK_TYPE_UNSPECIFIED = 0
    STANDARD = 1
    SSD = 2