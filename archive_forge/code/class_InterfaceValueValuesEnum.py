from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InterfaceValueValuesEnum(_messages.Enum):
    """Specifies the disk interface to use for attaching this disk, which is
    either SCSI or NVME.

    Values:
      NVME: <no description>
      SCSI: <no description>
    """
    NVME = 0
    SCSI = 1