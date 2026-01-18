from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SendDebugCaptureRequest(_messages.Message):
    """Request to send encoded debug information. Next ID: 8

  Enums:
    DataFormatValueValuesEnum: Format for the data field above (id=5).

  Fields:
    componentId: The internal component id for which debug information is
      sent.
    data: The encoded debug information.
    dataFormat: Format for the data field above (id=5).
    location: The [regional endpoint]
      (https://cloud.google.com/dataflow/docs/concepts/regional-endpoints)
      that contains the job specified by job_id.
    workerId: The worker id, i.e., VM hostname.
  """

    class DataFormatValueValuesEnum(_messages.Enum):
        """Format for the data field above (id=5).

    Values:
      DATA_FORMAT_UNSPECIFIED: Format unspecified, parsing is determined based
        upon page type and legacy encoding. (go/protodosdonts#do-include-an-
        unspecified-value-in-an-enum)
      RAW: Raw HTML string.
      JSON: JSON-encoded string.
      ZLIB: Websafe encoded zlib-compressed string.
      BROTLI: Websafe encoded brotli-compressed string.
    """
        DATA_FORMAT_UNSPECIFIED = 0
        RAW = 1
        JSON = 2
        ZLIB = 3
        BROTLI = 4
    componentId = _messages.StringField(1)
    data = _messages.StringField(2)
    dataFormat = _messages.EnumField('DataFormatValueValuesEnum', 3)
    location = _messages.StringField(4)
    workerId = _messages.StringField(5)