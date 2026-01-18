from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CompressionValueValuesEnum(_messages.Enum):
    """Compression of the loaded JSON file.

    Values:
      JSON_COMPRESSION_UNSPECIFIED: Unspecified json file compression.
      NO_COMPRESSION: Do not compress JSON file.
      GZIP: Gzip compression.
    """
    JSON_COMPRESSION_UNSPECIFIED = 0
    NO_COMPRESSION = 1
    GZIP = 2