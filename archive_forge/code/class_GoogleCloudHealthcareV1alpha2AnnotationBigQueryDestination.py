from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudHealthcareV1alpha2AnnotationBigQueryDestination(_messages.Message):
    """The BigQuery table for export.

  Enums:
    SchemaTypeValueValuesEnum: Specifies the schema format to export.
    WriteDispositionValueValuesEnum: Determines if existing data in the
      destination dataset is overwritten, appended to, or not written if the
      tables contain data. If a write_disposition is specified, the `force`
      parameter is ignored.

  Fields:
    force: Use `write_disposition` instead. If `write_disposition` is
      specified, this parameter is ignored. force=false is equivalent to
      write_disposition=WRITE_EMPTY and force=true is equivalent to
      write_disposition=WRITE_TRUNCATE.
    schemaType: Specifies the schema format to export.
    tableUri: BigQuery URI to a table, up to 2000 characters long, must be of
      the form bq://projectId.bqDatasetId.tableId.
    writeDisposition: Determines if existing data in the destination dataset
      is overwritten, appended to, or not written if the tables contain data.
      If a write_disposition is specified, the `force` parameter is ignored.
  """

    class SchemaTypeValueValuesEnum(_messages.Enum):
        """Specifies the schema format to export.

    Values:
      SCHEMA_TYPE_UNSPECIFIED: Same as SIMPLE.
      SIMPLE: A flatterned version of Annotation.
    """
        SCHEMA_TYPE_UNSPECIFIED = 0
        SIMPLE = 1

    class WriteDispositionValueValuesEnum(_messages.Enum):
        """Determines if existing data in the destination dataset is overwritten,
    appended to, or not written if the tables contain data. If a
    write_disposition is specified, the `force` parameter is ignored.

    Values:
      WRITE_DISPOSITION_UNSPECIFIED: Default behavior is the same as
        WRITE_EMPTY.
      WRITE_EMPTY: Only export data if the destination table is empty.
      WRITE_TRUNCATE: Erase all existing data in a table before writing the
        instances.
      WRITE_APPEND: Append data to the existing table.
    """
        WRITE_DISPOSITION_UNSPECIFIED = 0
        WRITE_EMPTY = 1
        WRITE_TRUNCATE = 2
        WRITE_APPEND = 3
    force = _messages.BooleanField(1)
    schemaType = _messages.EnumField('SchemaTypeValueValuesEnum', 2)
    tableUri = _messages.StringField(3)
    writeDisposition = _messages.EnumField('WriteDispositionValueValuesEnum', 4)