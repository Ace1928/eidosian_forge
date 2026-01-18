from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudHealthcareV1beta1DicomBigQueryDestination(_messages.Message):
    """The BigQuery table where the server writes output.

  Enums:
    WriteDispositionValueValuesEnum: Determines whether the existing table in
      the destination is to be overwritten or appended to. If a
      write_disposition is specified, the `force` parameter is ignored.

  Fields:
    force: Use `write_disposition` instead. If `write_disposition` is
      specified, this parameter is ignored. force=false is equivalent to
      write_disposition=WRITE_EMPTY and force=true is equivalent to
      write_disposition=WRITE_TRUNCATE.
    tableUri: BigQuery URI to a table, up to 2000 characters long, in the
      format `bq://projectId.bqDatasetId.tableId`
    writeDisposition: Determines whether the existing table in the destination
      is to be overwritten or appended to. If a write_disposition is
      specified, the `force` parameter is ignored.
  """

    class WriteDispositionValueValuesEnum(_messages.Enum):
        """Determines whether the existing table in the destination is to be
    overwritten or appended to. If a write_disposition is specified, the
    `force` parameter is ignored.

    Values:
      WRITE_DISPOSITION_UNSPECIFIED: Default behavior is the same as
        WRITE_EMPTY.
      WRITE_EMPTY: Only export data if the destination table is empty.
      WRITE_TRUNCATE: Erase all existing data in the destination table before
        writing the instances.
      WRITE_APPEND: Append data to the destination table.
    """
        WRITE_DISPOSITION_UNSPECIFIED = 0
        WRITE_EMPTY = 1
        WRITE_TRUNCATE = 2
        WRITE_APPEND = 3
    force = _messages.BooleanField(1)
    tableUri = _messages.StringField(2)
    writeDisposition = _messages.EnumField('WriteDispositionValueValuesEnum', 3)