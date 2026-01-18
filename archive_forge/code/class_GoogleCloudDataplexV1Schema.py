from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1Schema(_messages.Message):
    """Schema information describing the structure and layout of the data.

  Enums:
    PartitionStyleValueValuesEnum: Optional. The structure of paths containing
      partition data within the entity.

  Fields:
    fields: Optional. The sequence of fields describing data in table
      entities. Note: BigQuery SchemaFields are immutable.
    partitionFields: Optional. The sequence of fields describing the partition
      structure in entities. If this field is empty, there are no partitions
      within the data.
    partitionStyle: Optional. The structure of paths containing partition data
      within the entity.
    userManaged: Required. Set to true if user-managed or false if managed by
      Dataplex. The default is false (managed by Dataplex). Set to falseto
      enable Dataplex discovery to update the schema. including new data
      discovery, schema inference, and schema evolution. Users retain the
      ability to input and edit the schema. Dataplex treats schema input by
      the user as though produced by a previous Dataplex discovery operation,
      and it will evolve the schema and take action based on that treatment.
      Set to true to fully manage the entity schema. This setting guarantees
      that Dataplex will not change schema fields.
  """

    class PartitionStyleValueValuesEnum(_messages.Enum):
        """Optional. The structure of paths containing partition data within the
    entity.

    Values:
      PARTITION_STYLE_UNSPECIFIED: PartitionStyle unspecified
      HIVE_COMPATIBLE: Partitions are hive-compatible. Examples:
        gs://bucket/path/to/table/dt=2019-10-31/lang=en,
        gs://bucket/path/to/table/dt=2019-10-31/lang=en/late.
    """
        PARTITION_STYLE_UNSPECIFIED = 0
        HIVE_COMPATIBLE = 1
    fields = _messages.MessageField('GoogleCloudDataplexV1SchemaSchemaField', 1, repeated=True)
    partitionFields = _messages.MessageField('GoogleCloudDataplexV1SchemaPartitionField', 2, repeated=True)
    partitionStyle = _messages.EnumField('PartitionStyleValueValuesEnum', 3)
    userManaged = _messages.BooleanField(4)