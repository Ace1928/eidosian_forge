from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GcsFileFormatValueValuesEnum(_messages.Enum):
    """File format that data should be written in. Deprecated field
    (b/169501737) - use file_format instead.

    Values:
      GCS_FILE_FORMAT_UNSPECIFIED: Unspecified Cloud Storage file format.
      AVRO: Avro file format
    """
    GCS_FILE_FORMAT_UNSPECIFIED = 0
    AVRO = 1