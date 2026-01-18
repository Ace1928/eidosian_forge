from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1Entry(_messages.Message):
    """Entry metadata. A Data Catalog entry represents another resource in
  Google Cloud Platform (such as a BigQuery dataset or a Pub/Sub topic) or
  outside of it. You can use the `linked_resource` field in the entry resource
  to refer to the original resource ID of the source system. An entry resource
  contains resource details, for example, its schema. Additionally, you can
  attach flexible metadata to an entry in the form of a Tag.

  Enums:
    IntegratedSystemValueValuesEnum: Output only. Indicates the entry's source
      system that Data Catalog integrates with, such as BigQuery, Pub/Sub, or
      Dataproc Metastore.
    TypeValueValuesEnum: The type of the entry. For details, see
      [`EntryType`](#entrytype).

  Messages:
    LabelsValue: Cloud labels attached to the entry. In Data Catalog, you can
      create and modify labels attached only to custom entries. Synced entries
      have unmodifiable labels that come from the source system.

  Fields:
    bigqueryDateShardedSpec: Output only. Specification for a group of
      BigQuery tables with the `[prefix]YYYYMMDD` name pattern. For more
      information, see [Introduction to partitioned tables]
      (https://cloud.google.com/bigquery/docs/partitioned-
      tables#partitioning_versus_sharding).
    bigqueryTableSpec: Output only. Specification that applies to a BigQuery
      table. Valid only for entries with the `TABLE` type.
    businessContext: Business Context of the entry. Not supported for BigQuery
      datasets
    cloudBigtableSystemSpec: Specification that applies to Cloud Bigtable
      system. Only settable when `integrated_system` is equal to
      `CLOUD_BIGTABLE`
    clusterSpec: Additional specification of a cluster. Present only on the
      entries that represent clusters.
    dataSource: Output only. Physical location of the entry.
    dataSourceConnectionSpec: Specification that applies to a data source
      connection. Valid only for entries with the `DATA_SOURCE_CONNECTION`
      type.
    dataStreamSpec: Additional specification of a non-Pub/Sub data stream.
    databaseTableSpec: Specification that applies to a table resource. Valid
      only for entries with the `TABLE` or `EXPLORE` type.
    datasetSpec: Specification that applies to a dataset.
    description: Entry description that can consist of several sentences or
      paragraphs that describe entry contents. The description must not
      contain Unicode non-characters as well as C0 and C1 control codes except
      tabs (HT), new lines (LF), carriage returns (CR), and page breaks (FF).
      The maximum size is 2000 bytes when encoded in UTF-8. Default value is
      an empty string.
    displayName: Display name of an entry. The maximum size is 500 bytes when
      encoded in UTF-8. Default value is an empty string.
    featureOnlineStoreSpec: FeatureonlineStore spec for Vertex AI Feature
      Store.
    filesetSpec: Specification that applies to a fileset resource. Valid only
      for entries with the `FILESET` type.
    fullyQualifiedName: [Fully Qualified Name
      (FQN)](https://cloud.google.com//data-catalog/docs/fully-qualified-
      names) of the resource. Set automatically for entries representing
      resources from synced systems. Settable only during creation, and read-
      only later. Can be used for search and lookup of the entries.
    gcsFilesetSpec: Specification that applies to a Cloud Storage fileset.
      Valid only for entries with the `FILESET` type.
    integratedSystem: Output only. Indicates the entry's source system that
      Data Catalog integrates with, such as BigQuery, Pub/Sub, or Dataproc
      Metastore.
    labels: Cloud labels attached to the entry. In Data Catalog, you can
      create and modify labels attached only to custom entries. Synced entries
      have unmodifiable labels that come from the source system.
    linkedResource: The resource this metadata entry refers to. For Google
      Cloud Platform resources, `linked_resource` is the [Full Resource Name]
      (https://cloud.google.com/apis/design/resource_names#full_resource_name)
      . For example, the `linked_resource` for a table resource from BigQuery
      is: `//bigquery.googleapis.com/projects/{PROJECT_ID}/datasets/{DATASET_I
      D}/tables/{TABLE_ID}` Output only when the entry is one of the types in
      the `EntryType` enum. For entries with a `user_specified_type`, this
      field is optional and defaults to an empty string. The resource string
      must contain only letters (a-z, A-Z), numbers (0-9), underscores (_),
      periods (.), colons (:), slashes (/), dashes (-), and hashes (#). The
      maximum size is 200 bytes when encoded in UTF-8.
    lookerSystemSpec: Specification that applies to Looker sysstem. Only
      settable when `user_specified_system` is equal to `LOOKER`
    modelSpec: Model specification.
    name: Output only. Identifier. The resource name of an entry in URL
      format. Note: The entry itself and its child resources might not be
      stored in the location specified in its name.
    personalDetails: Output only. Additional information related to the entry.
      Private to the current user.
    routineSpec: Specification that applies to a user-defined function or
      procedure. Valid only for entries with the `ROUTINE` type.
    schema: Schema of the entry. An entry might not have any schema attached
      to it.
    serviceSpec: Specification that applies to a Service resource.
    sourceSystemTimestamps: Timestamps from the underlying resource, not from
      the Data Catalog entry. Output only when the entry has a system listed
      in the `IntegratedSystem` enum. For entries with
      `user_specified_system`, this field is optional and defaults to an empty
      timestamp.
    sqlDatabaseSystemSpec: Specification that applies to a relational database
      system. Only settable when `user_specified_system` is equal to
      `SQL_DATABASE`
    type: The type of the entry. For details, see [`EntryType`](#entrytype).
    usageSignal: Resource usage statistics.
    userSpecifiedSystem: Indicates the entry's source system that Data Catalog
      doesn't automatically integrate with. The `user_specified_system` string
      has the following limitations: * Is case insensitive. * Must begin with
      a letter or underscore. * Can only contain letters, numbers, and
      underscores. * Must be at least 1 character and at most 64 characters
      long.
    userSpecifiedType: Custom entry type that doesn't match any of the values
      allowed for input and listed in the `EntryType` enum. When creating an
      entry, first check the type values in the enum. If there are no
      appropriate types for the new entry, provide a custom value, for
      example, `my_special_type`. The `user_specified_type` string has the
      following limitations: * Is case insensitive. * Must begin with a letter
      or underscore. * Can only contain letters, numbers, and underscores. *
      Must be at least 1 character and at most 64 characters long.
  """

    class IntegratedSystemValueValuesEnum(_messages.Enum):
        """Output only. Indicates the entry's source system that Data Catalog
    integrates with, such as BigQuery, Pub/Sub, or Dataproc Metastore.

    Values:
      INTEGRATED_SYSTEM_UNSPECIFIED: Default unknown system.
      BIGQUERY: BigQuery.
      CLOUD_PUBSUB: Cloud Pub/Sub.
      DATAPROC_METASTORE: Dataproc Metastore.
      DATAPLEX: Dataplex.
      CLOUD_SPANNER: Cloud Spanner
      CLOUD_BIGTABLE: Cloud Bigtable
      CLOUD_SQL: Cloud Sql
      LOOKER: Looker
      VERTEX_AI: Vertex AI
    """
        INTEGRATED_SYSTEM_UNSPECIFIED = 0
        BIGQUERY = 1
        CLOUD_PUBSUB = 2
        DATAPROC_METASTORE = 3
        DATAPLEX = 4
        CLOUD_SPANNER = 5
        CLOUD_BIGTABLE = 6
        CLOUD_SQL = 7
        LOOKER = 8
        VERTEX_AI = 9

    class TypeValueValuesEnum(_messages.Enum):
        """The type of the entry. For details, see [`EntryType`](#entrytype).

    Values:
      ENTRY_TYPE_UNSPECIFIED: Default unknown type.
      TABLE: The entry type that has a GoogleSQL schema, including logical
        views.
      MODEL: The type of models. For more information, see [Supported models
        in BigQuery ML](/bigquery/docs/bqml-introduction#supported_models).
      DATA_STREAM: An entry type for streaming entries. For example, a Pub/Sub
        topic.
      FILESET: An entry type for a set of files or objects. For example, a
        Cloud Storage fileset.
      CLUSTER: A group of servers that work together. For example, a Kafka
        cluster.
      DATABASE: A database.
      DATA_SOURCE_CONNECTION: Connection to a data source. For example, a
        BigQuery connection.
      ROUTINE: Routine, for example, a BigQuery routine.
      LAKE: A Dataplex lake.
      ZONE: A Dataplex zone.
      SERVICE: A service, for example, a Dataproc Metastore service.
      DATABASE_SCHEMA: Schema within a relational database.
      DASHBOARD: A Dashboard, for example from Looker.
      EXPLORE: A Looker Explore. For more information, see [Looker Explore
        API] (https://developers.looker.com/api/explorer/4.0/methods/LookmlMod
        el/lookml_model_explore).
      LOOK: A Looker Look. For more information, see [Looker Look API]
        (https://developers.looker.com/api/explorer/4.0/methods/Look).
      FEATURE_ONLINE_STORE: Feature Online Store resource in Vertex AI Feature
        Store.
      FEATURE_VIEW: Feature View resource in Vertex AI Feature Store.
      FEATURE_GROUP: Feature Group resource in Vertex AI Feature Store.
    """
        ENTRY_TYPE_UNSPECIFIED = 0
        TABLE = 1
        MODEL = 2
        DATA_STREAM = 3
        FILESET = 4
        CLUSTER = 5
        DATABASE = 6
        DATA_SOURCE_CONNECTION = 7
        ROUTINE = 8
        LAKE = 9
        ZONE = 10
        SERVICE = 11
        DATABASE_SCHEMA = 12
        DASHBOARD = 13
        EXPLORE = 14
        LOOK = 15
        FEATURE_ONLINE_STORE = 16
        FEATURE_VIEW = 17
        FEATURE_GROUP = 18

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Cloud labels attached to the entry. In Data Catalog, you can create
    and modify labels attached only to custom entries. Synced entries have
    unmodifiable labels that come from the source system.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    bigqueryDateShardedSpec = _messages.MessageField('GoogleCloudDatacatalogV1BigQueryDateShardedSpec', 1)
    bigqueryTableSpec = _messages.MessageField('GoogleCloudDatacatalogV1BigQueryTableSpec', 2)
    businessContext = _messages.MessageField('GoogleCloudDatacatalogV1BusinessContext', 3)
    cloudBigtableSystemSpec = _messages.MessageField('GoogleCloudDatacatalogV1CloudBigtableSystemSpec', 4)
    clusterSpec = _messages.MessageField('GoogleCloudDatacatalogV1ClusterSpec', 5)
    dataSource = _messages.MessageField('GoogleCloudDatacatalogV1DataSource', 6)
    dataSourceConnectionSpec = _messages.MessageField('GoogleCloudDatacatalogV1DataSourceConnectionSpec', 7)
    dataStreamSpec = _messages.MessageField('GoogleCloudDatacatalogV1DataStreamSpec', 8)
    databaseTableSpec = _messages.MessageField('GoogleCloudDatacatalogV1DatabaseTableSpec', 9)
    datasetSpec = _messages.MessageField('GoogleCloudDatacatalogV1DatasetSpec', 10)
    description = _messages.StringField(11)
    displayName = _messages.StringField(12)
    featureOnlineStoreSpec = _messages.MessageField('GoogleCloudDatacatalogV1FeatureOnlineStoreSpec', 13)
    filesetSpec = _messages.MessageField('GoogleCloudDatacatalogV1FilesetSpec', 14)
    fullyQualifiedName = _messages.StringField(15)
    gcsFilesetSpec = _messages.MessageField('GoogleCloudDatacatalogV1GcsFilesetSpec', 16)
    integratedSystem = _messages.EnumField('IntegratedSystemValueValuesEnum', 17)
    labels = _messages.MessageField('LabelsValue', 18)
    linkedResource = _messages.StringField(19)
    lookerSystemSpec = _messages.MessageField('GoogleCloudDatacatalogV1LookerSystemSpec', 20)
    modelSpec = _messages.MessageField('GoogleCloudDatacatalogV1ModelSpec', 21)
    name = _messages.StringField(22)
    personalDetails = _messages.MessageField('GoogleCloudDatacatalogV1PersonalDetails', 23)
    routineSpec = _messages.MessageField('GoogleCloudDatacatalogV1RoutineSpec', 24)
    schema = _messages.MessageField('GoogleCloudDatacatalogV1Schema', 25)
    serviceSpec = _messages.MessageField('GoogleCloudDatacatalogV1ServiceSpec', 26)
    sourceSystemTimestamps = _messages.MessageField('GoogleCloudDatacatalogV1SystemTimestamps', 27)
    sqlDatabaseSystemSpec = _messages.MessageField('GoogleCloudDatacatalogV1SqlDatabaseSystemSpec', 28)
    type = _messages.EnumField('TypeValueValuesEnum', 29)
    usageSignal = _messages.MessageField('GoogleCloudDatacatalogV1UsageSignal', 30)
    userSpecifiedSystem = _messages.StringField(31)
    userSpecifiedType = _messages.StringField(32)