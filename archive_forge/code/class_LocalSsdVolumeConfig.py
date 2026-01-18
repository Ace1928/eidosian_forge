from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LocalSsdVolumeConfig(_messages.Message):
    """LocalSsdVolumeConfig is composed of three fields, count, type, and
  format. Count is the number of ssds of this grouping requested, type is the
  interface type and is either nvme or scsi, and format is whether the disk is
  to be formatted with a filesystem or left for block storage

  Enums:
    FormatValueValuesEnum: Format of the local SSD (fs/block).

  Fields:
    count: Number of local SSDs to use
    format: Format of the local SSD (fs/block).
    type: Local SSD interface to use (nvme/scsi).
  """

    class FormatValueValuesEnum(_messages.Enum):
        """Format of the local SSD (fs/block).

    Values:
      FORMAT_UNSPECIFIED: Default value
      FS: File system formatted
      BLOCK: Raw block
    """
        FORMAT_UNSPECIFIED = 0
        FS = 1
        BLOCK = 2
    count = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    format = _messages.EnumField('FormatValueValuesEnum', 2)
    type = _messages.StringField(3)