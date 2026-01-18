from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MetadataExport(_messages.Message):
    """The details of a metadata export operation.

  Enums:
    DatabaseDumpTypeValueValuesEnum: Output only. The type of the database
      dump.
    StateValueValuesEnum: Output only. The current state of the export.

  Fields:
    databaseDumpType: Output only. The type of the database dump.
    destinationGcsUri: Output only. A Cloud Storage URI of a folder that
      metadata are exported to, in the form of gs:////, where is automatically
      generated.
    endTime: Output only. The time when the export ended.
    startTime: Output only. The time when the export started.
    state: Output only. The current state of the export.
  """

    class DatabaseDumpTypeValueValuesEnum(_messages.Enum):
        """Output only. The type of the database dump.

    Values:
      TYPE_UNSPECIFIED: The type of the database dump is unknown.
      MYSQL: Database dump is a MySQL dump file.
      AVRO: Database dump contains Avro files.
    """
        TYPE_UNSPECIFIED = 0
        MYSQL = 1
        AVRO = 2

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The current state of the export.

    Values:
      STATE_UNSPECIFIED: The state of the metadata export is unknown.
      RUNNING: The metadata export is running.
      SUCCEEDED: The metadata export completed successfully.
      FAILED: The metadata export failed.
      CANCELLED: The metadata export is cancelled.
    """
        STATE_UNSPECIFIED = 0
        RUNNING = 1
        SUCCEEDED = 2
        FAILED = 3
        CANCELLED = 4
    databaseDumpType = _messages.EnumField('DatabaseDumpTypeValueValuesEnum', 1)
    destinationGcsUri = _messages.StringField(2)
    endTime = _messages.StringField(3)
    startTime = _messages.StringField(4)
    state = _messages.EnumField('StateValueValuesEnum', 5)