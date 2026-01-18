from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TableMetadataCacheUsage(_messages.Message):
    """Table level detail on the usage of metadata caching. Only set for
  Metadata caching eligible tables referenced in the query.

  Enums:
    UnusedReasonValueValuesEnum: Reason for not using metadata caching for the
      table.

  Fields:
    explanation: Free form human-readable reason metadata caching was unused
      for the job.
    tableReference: Metadata caching eligible table referenced in the query.
    tableType: [Table
      type](/bigquery/docs/reference/rest/v2/tables#Table.FIELDS.type).
    unusedReason: Reason for not using metadata caching for the table.
  """

    class UnusedReasonValueValuesEnum(_messages.Enum):
        """Reason for not using metadata caching for the table.

    Values:
      UNUSED_REASON_UNSPECIFIED: Unused reasons not specified.
      EXCEEDED_MAX_STALENESS: Metadata cache was outside the table's
        maxStaleness.
      METADATA_CACHING_NOT_ENABLED: Metadata caching feature is not enabled.
        [Update BigLake tables] (/bigquery/docs/create-cloud-storage-table-
        biglake#update-biglake-tables) to enable the metadata caching.
      OTHER_REASON: Other unknown reason.
    """
        UNUSED_REASON_UNSPECIFIED = 0
        EXCEEDED_MAX_STALENESS = 1
        METADATA_CACHING_NOT_ENABLED = 2
        OTHER_REASON = 3
    explanation = _messages.StringField(1)
    tableReference = _messages.MessageField('TableReference', 2)
    tableType = _messages.StringField(3)
    unusedReason = _messages.EnumField('UnusedReasonValueValuesEnum', 4)