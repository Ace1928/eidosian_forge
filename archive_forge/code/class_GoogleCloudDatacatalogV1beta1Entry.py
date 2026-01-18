from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1beta1Entry(_messages.Message):
    """Entry Metadata. A Data Catalog Entry resource represents another
  resource in Google Cloud Platform (such as a BigQuery dataset or a Pub/Sub
  topic), or outside of Google Cloud Platform. Clients can use the
  `linked_resource` field in the Entry resource to refer to the original
  resource ID of the source system. An Entry resource contains resource
  details, such as its schema. An Entry can also be used to attach flexible
  metadata, such as a Tag.

  Enums:
    IntegratedSystemValueValuesEnum: Output only. This field indicates the
      entry's source system that Data Catalog integrates with, such as
      BigQuery or Pub/Sub.
    TypeValueValuesEnum: The type of the entry. Only used for Entries with
      types in the EntryType enum.

  Fields:
    bigqueryDateShardedSpec: Specification for a group of BigQuery tables with
      name pattern `[prefix]YYYYMMDD`. Context:
      https://cloud.google.com/bigquery/docs/partitioned-
      tables#partitioning_versus_sharding.
    bigqueryTableSpec: Specification that applies to a BigQuery table. This is
      only valid on entries of type `TABLE`.
    description: Entry description, which can consist of several sentences or
      paragraphs that describe entry contents. Default value is an empty
      string.
    displayName: Display information such as title and description. A short
      name to identify the entry, for example, "Analytics Data - Jan 2011".
      Default value is an empty string.
    gcsFilesetSpec: Specification that applies to a Cloud Storage fileset.
      This is only valid on entries of type FILESET.
    integratedSystem: Output only. This field indicates the entry's source
      system that Data Catalog integrates with, such as BigQuery or Pub/Sub.
    linkedResource: The resource this metadata entry refers to. For Google
      Cloud Platform resources, `linked_resource` is the [full name of the res
      ource](https://cloud.google.com/apis/design/resource_names#full_resource
      _name). For example, the `linked_resource` for a table resource from
      BigQuery is: * //bigquery.googleapis.com/projects/projectId/datasets/dat
      asetId/tables/tableId Output only when Entry is of type in the EntryType
      enum. For entries with user_specified_type, this field is optional and
      defaults to an empty string.
    name: Output only. Identifier. The Data Catalog resource name of the entry
      in URL format. Example: * projects/{project_id}/locations/{location}/ent
      ryGroups/{entry_group_id}/entries/{entry_id} Note that this Entry and
      its child resources may not actually be stored in the location in this
      name.
    schema: Schema of the entry. An entry might not have any schema attached
      to it.
    sourceSystemTimestamps: Output only. Timestamps about the underlying
      resource, not about this Data Catalog entry. Output only when Entry is
      of type in the EntryType enum. For entries with user_specified_type,
      this field is optional and defaults to an empty timestamp.
    type: The type of the entry. Only used for Entries with types in the
      EntryType enum.
    usageSignal: Output only. Statistics on the usage level of the resource.
    userSpecifiedSystem: This field indicates the entry's source system that
      Data Catalog does not integrate with. `user_specified_system` strings
      must begin with a letter or underscore and can only contain letters,
      numbers, and underscores; are case insensitive; must be at least 1
      character and at most 64 characters long.
    userSpecifiedType: Entry type if it does not fit any of the input-allowed
      values listed in `EntryType` enum above. When creating an entry, users
      should check the enum values first, if nothing matches the entry to be
      created, then provide a custom value, for example "my_special_type".
      `user_specified_type` strings must begin with a letter or underscore and
      can only contain letters, numbers, and underscores; are case
      insensitive; must be at least 1 character and at most 64 characters
      long. Currently, only FILESET enum value is allowed. All other entries
      created through Data Catalog must use `user_specified_type`.
  """

    class IntegratedSystemValueValuesEnum(_messages.Enum):
        """Output only. This field indicates the entry's source system that Data
    Catalog integrates with, such as BigQuery or Pub/Sub.

    Values:
      INTEGRATED_SYSTEM_UNSPECIFIED: Default unknown system.
      BIGQUERY: BigQuery.
      CLOUD_PUBSUB: Cloud Pub/Sub.
    """
        INTEGRATED_SYSTEM_UNSPECIFIED = 0
        BIGQUERY = 1
        CLOUD_PUBSUB = 2

    class TypeValueValuesEnum(_messages.Enum):
        """The type of the entry. Only used for Entries with types in the
    EntryType enum.

    Values:
      ENTRY_TYPE_UNSPECIFIED: Default unknown type.
      TABLE: Output only. The type of entry that has a GoogleSQL schema,
        including logical views.
      MODEL: Output only. The type of models.
        https://cloud.google.com/bigquery-ml/docs/bigqueryml-intro
      DATA_STREAM: Output only. An entry type which is used for streaming
        entries. Example: Pub/Sub topic.
      FILESET: An entry type which is a set of files or objects. Example:
        Cloud Storage fileset.
    """
        ENTRY_TYPE_UNSPECIFIED = 0
        TABLE = 1
        MODEL = 2
        DATA_STREAM = 3
        FILESET = 4
    bigqueryDateShardedSpec = _messages.MessageField('GoogleCloudDatacatalogV1beta1BigQueryDateShardedSpec', 1)
    bigqueryTableSpec = _messages.MessageField('GoogleCloudDatacatalogV1beta1BigQueryTableSpec', 2)
    description = _messages.StringField(3)
    displayName = _messages.StringField(4)
    gcsFilesetSpec = _messages.MessageField('GoogleCloudDatacatalogV1beta1GcsFilesetSpec', 5)
    integratedSystem = _messages.EnumField('IntegratedSystemValueValuesEnum', 6)
    linkedResource = _messages.StringField(7)
    name = _messages.StringField(8)
    schema = _messages.MessageField('GoogleCloudDatacatalogV1beta1Schema', 9)
    sourceSystemTimestamps = _messages.MessageField('GoogleCloudDatacatalogV1beta1SystemTimestamps', 10)
    type = _messages.EnumField('TypeValueValuesEnum', 11)
    usageSignal = _messages.MessageField('GoogleCloudDatacatalogV1beta1UsageSignal', 12)
    userSpecifiedSystem = _messages.StringField(13)
    userSpecifiedType = _messages.StringField(14)