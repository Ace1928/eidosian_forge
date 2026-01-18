from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2TimespanConfig(_messages.Message):
    """Configuration of the timespan of the items to include in scanning.
  Currently only supported when inspecting Cloud Storage and BigQuery.

  Fields:
    enableAutoPopulationOfTimespanConfig: When the job is started by a
      JobTrigger we will automatically figure out a valid start_time to avoid
      scanning files that have not been modified since the last time the
      JobTrigger executed. This will be based on the time of the execution of
      the last run of the JobTrigger or the timespan end_time used in the last
      run of the JobTrigger.
    endTime: Exclude files, tables, or rows newer than this value. If not set,
      no upper time limit is applied.
    startTime: Exclude files, tables, or rows older than this value. If not
      set, no lower time limit is applied.
    timestampField: Specification of the field containing the timestamp of
      scanned items. Used for data sources like Datastore and BigQuery. *For
      BigQuery* If this value is not specified and the table was modified
      between the given start and end times, the entire table will be scanned.
      If this value is specified, then rows are filtered based on the given
      start and end times. Rows with a `NULL` value in the provided BigQuery
      column are skipped. Valid data types of the provided BigQuery column
      are: `INTEGER`, `DATE`, `TIMESTAMP`, and `DATETIME`. If your BigQuery
      table is [partitioned at ingestion
      time](https://cloud.google.com/bigquery/docs/partitioned-
      tables#ingestion_time), you can use any of the following pseudo-columns
      as your timestamp field. When used with Cloud DLP, these pseudo-column
      names are case sensitive. - _PARTITIONTIME - _PARTITIONDATE -
      _PARTITION_LOAD_TIME *For Datastore* If this value is specified, then
      entities are filtered based on the given start and end times. If an
      entity does not contain the provided timestamp property or contains
      empty or invalid values, then it is included. Valid data types of the
      provided timestamp property are: `TIMESTAMP`. See the [known
      issue](https://cloud.google.com/sensitive-data-protection/docs/known-
      issues#bq-timespan) related to this operation.
  """
    enableAutoPopulationOfTimespanConfig = _messages.BooleanField(1)
    endTime = _messages.StringField(2)
    startTime = _messages.StringField(3)
    timestampField = _messages.MessageField('GooglePrivacyDlpV2FieldId', 4)