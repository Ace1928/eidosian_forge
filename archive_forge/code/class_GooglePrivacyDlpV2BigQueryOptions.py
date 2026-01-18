from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2BigQueryOptions(_messages.Message):
    """Options defining BigQuery table and row identifiers.

  Enums:
    SampleMethodValueValuesEnum: How to sample the data.

  Fields:
    excludedFields: References to fields excluded from scanning. This allows
      you to skip inspection of entire columns which you know have no
      findings. When inspecting a table, we recommend that you inspect all
      columns. Otherwise, findings might be affected because hints from
      excluded columns will not be used.
    identifyingFields: Table fields that may uniquely identify a row within
      the table. When `actions.saveFindings.outputConfig.table` is specified,
      the values of columns specified here are available in the output table
      under `location.content_locations.record_location.record_key.id_values`.
      Nested fields such as `person.birthdate.year` are allowed.
    includedFields: Limit scanning only to these fields. When inspecting a
      table, we recommend that you inspect all columns. Otherwise, findings
      might be affected because hints from excluded columns will not be used.
    rowsLimit: Max number of rows to scan. If the table has more rows than
      this value, the rest of the rows are omitted. If not set, or if set to
      0, all rows will be scanned. Only one of rows_limit and
      rows_limit_percent can be specified. Cannot be used in conjunction with
      TimespanConfig.
    rowsLimitPercent: Max percentage of rows to scan. The rest are omitted.
      The number of rows scanned is rounded down. Must be between 0 and 100,
      inclusively. Both 0 and 100 means no limit. Defaults to 0. Only one of
      rows_limit and rows_limit_percent can be specified. Cannot be used in
      conjunction with TimespanConfig. Caution: A [known
      issue](https://cloud.google.com/sensitive-data-protection/docs/known-
      issues#bq-sampling) is causing the `rowsLimitPercent` field to behave
      unexpectedly. We recommend using `rowsLimit` instead.
    sampleMethod: How to sample the data.
    tableReference: Complete BigQuery table reference.
  """

    class SampleMethodValueValuesEnum(_messages.Enum):
        """How to sample the data.

    Values:
      SAMPLE_METHOD_UNSPECIFIED: No sampling.
      TOP: Scan groups of rows in the order BigQuery provides (default).
        Multiple groups of rows may be scanned in parallel, so results may not
        appear in the same order the rows are read.
      RANDOM_START: Randomly pick groups of rows to scan.
    """
        SAMPLE_METHOD_UNSPECIFIED = 0
        TOP = 1
        RANDOM_START = 2
    excludedFields = _messages.MessageField('GooglePrivacyDlpV2FieldId', 1, repeated=True)
    identifyingFields = _messages.MessageField('GooglePrivacyDlpV2FieldId', 2, repeated=True)
    includedFields = _messages.MessageField('GooglePrivacyDlpV2FieldId', 3, repeated=True)
    rowsLimit = _messages.IntegerField(4)
    rowsLimitPercent = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    sampleMethod = _messages.EnumField('SampleMethodValueValuesEnum', 6)
    tableReference = _messages.MessageField('GooglePrivacyDlpV2BigQueryTable', 7)