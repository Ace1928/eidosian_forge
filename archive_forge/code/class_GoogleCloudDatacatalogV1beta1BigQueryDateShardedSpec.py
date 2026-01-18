from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1beta1BigQueryDateShardedSpec(_messages.Message):
    """Spec for a group of BigQuery tables with name pattern
  `[prefix]YYYYMMDD`. Context:
  https://cloud.google.com/bigquery/docs/partitioned-
  tables#partitioning_versus_sharding

  Fields:
    dataset: Output only. The Data Catalog resource name of the dataset entry
      the current table belongs to, for example, `projects/{project_id}/locati
      ons/{location}/entrygroups/{entry_group_id}/entries/{entry_id}`.
    shardCount: Output only. Total number of shards.
    tablePrefix: Output only. The table name prefix of the shards. The name of
      any given shard is `[table_prefix]YYYYMMDD`, for example, for shard
      `MyTable20180101`, the `table_prefix` is `MyTable`.
  """
    dataset = _messages.StringField(1)
    shardCount = _messages.IntegerField(2)
    tablePrefix = _messages.StringField(3)