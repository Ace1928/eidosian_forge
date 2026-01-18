from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TelemetryConfig(_messages.Message):
    """Telemetry Configuration for the Dataproc Metastore service.

  Enums:
    LogFormatValueValuesEnum: The output format of the Dataproc Metastore
      service's logs.

  Fields:
    logFormat: The output format of the Dataproc Metastore service's logs.
  """

    class LogFormatValueValuesEnum(_messages.Enum):
        """The output format of the Dataproc Metastore service's logs.

    Values:
      LOG_FORMAT_UNSPECIFIED: The LOG_FORMAT is not set.
      LEGACY: Logging output uses the legacy textPayload format.
      JSON: Logging output uses the jsonPayload format.
    """
        LOG_FORMAT_UNSPECIFIED = 0
        LEGACY = 1
        JSON = 2
    logFormat = _messages.EnumField('LogFormatValueValuesEnum', 1)