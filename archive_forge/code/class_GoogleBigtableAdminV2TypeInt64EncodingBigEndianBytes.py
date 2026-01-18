from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleBigtableAdminV2TypeInt64EncodingBigEndianBytes(_messages.Message):
    """Encodes the value as an 8-byte big endian twos complement `Bytes` value.
  * Natural sort? No (positive values only) * Self-delimiting? Yes *
  Compatibility? - BigQuery Federation `BINARY` encoding - HBase
  `Bytes.toBytes` - Java `ByteBuffer.putLong()` with `ByteOrder.BIG_ENDIAN`

  Fields:
    bytesType: The underlying `Bytes` type, which may be able to encode
      further.
  """
    bytesType = _messages.MessageField('GoogleBigtableAdminV2TypeBytes', 1)