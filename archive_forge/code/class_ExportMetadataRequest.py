from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExportMetadataRequest(_messages.Message):
    """Request message for DataprocMetastore.ExportMetadata.

  Enums:
    DatabaseDumpTypeValueValuesEnum: Optional. The type of the database dump.
      If unspecified, defaults to MYSQL.

  Fields:
    databaseDumpType: Optional. The type of the database dump. If unspecified,
      defaults to MYSQL.
    destinationGcsFolder: A Cloud Storage URI of a folder, in the format
      gs:///. A sub-folder containing exported files will be created below it.
    requestId: Optional. A request ID. Specify a unique request ID to allow
      the server to ignore the request if it has completed. The server will
      ignore subsequent requests that provide a duplicate request ID for at
      least 60 minutes after the first request.For example, if an initial
      request times out, followed by another request with the same request ID,
      the server ignores the second request to prevent the creation of
      duplicate commitments.The request ID must be a valid UUID
      (https://en.wikipedia.org/wiki/Universally_unique_identifier#Format). A
      zero UUID (00000000-0000-0000-0000-000000000000) is not supported.
  """

    class DatabaseDumpTypeValueValuesEnum(_messages.Enum):
        """Optional. The type of the database dump. If unspecified, defaults to
    MYSQL.

    Values:
      TYPE_UNSPECIFIED: The type of the database dump is unknown.
      MYSQL: Database dump is a MySQL dump file.
      AVRO: Database dump contains Avro files.
    """
        TYPE_UNSPECIFIED = 0
        MYSQL = 1
        AVRO = 2
    databaseDumpType = _messages.EnumField('DatabaseDumpTypeValueValuesEnum', 1)
    destinationGcsFolder = _messages.StringField(2)
    requestId = _messages.StringField(3)