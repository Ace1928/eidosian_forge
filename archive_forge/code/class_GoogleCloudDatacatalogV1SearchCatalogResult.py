from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1SearchCatalogResult(_messages.Message):
    """Result in the response to a search request. Each result captures details
  of one entry that matches the search.

  Enums:
    IntegratedSystemValueValuesEnum: Output only. The source system that Data
      Catalog automatically integrates with, such as BigQuery, Cloud Pub/Sub,
      or Dataproc Metastore.
    SearchResultTypeValueValuesEnum: Type of the search result. You can use
      this field to determine which get method to call to fetch the full
      resource.

  Fields:
    description: Entry description that can consist of several sentences or
      paragraphs that describe entry contents.
    displayName: The display name of the result.
    fullyQualifiedName: Fully qualified name (FQN) of the resource. FQNs take
      two forms: * For non-regionalized resources:
      `{SYSTEM}:{PROJECT}.{PATH_TO_RESOURCE_SEPARATED_WITH_DOTS}` * For
      regionalized resources: `{SYSTEM}:{PROJECT}.{LOCATION_ID}.{PATH_TO_RESOU
      RCE_SEPARATED_WITH_DOTS}` Example for a DPMS table: `dataproc_metastore:
      PROJECT_ID.LOCATION_ID.INSTANCE_ID.DATABASE_ID.TABLE_ID`
    integratedSystem: Output only. The source system that Data Catalog
      automatically integrates with, such as BigQuery, Cloud Pub/Sub, or
      Dataproc Metastore.
    linkedResource: The full name of the Google Cloud resource the entry
      belongs to. For more information, see [Full Resource Name]
      (/apis/design/resource_names#full_resource_name). Example: `//bigquery.g
      oogleapis.com/projects/PROJECT_ID/datasets/DATASET_ID/tables/TABLE_ID`
    modifyTime: The last modification timestamp of the entry in the source
      system.
    relativeResourceName: The relative name of the resource in URL format.
      Examples: * `projects/{PROJECT_ID}/locations/{LOCATION_ID}/entryGroups/{
      ENTRY_GROUP_ID}/entries/{ENTRY_ID}` *
      `projects/{PROJECT_ID}/tagTemplates/{TAG_TEMPLATE_ID}`
    searchResultSubtype: Sub-type of the search result. A dot-delimited full
      type of the resource. The same type you specify in the `type` search
      predicate. Examples: `entry.table`, `entry.dataStream`, `tagTemplate`.
    searchResultType: Type of the search result. You can use this field to
      determine which get method to call to fetch the full resource.
    userSpecifiedSystem: Custom source system that you can manually integrate
      Data Catalog with.
  """

    class IntegratedSystemValueValuesEnum(_messages.Enum):
        """Output only. The source system that Data Catalog automatically
    integrates with, such as BigQuery, Cloud Pub/Sub, or Dataproc Metastore.

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

    class SearchResultTypeValueValuesEnum(_messages.Enum):
        """Type of the search result. You can use this field to determine which
    get method to call to fetch the full resource.

    Values:
      SEARCH_RESULT_TYPE_UNSPECIFIED: Default unknown type.
      ENTRY: An Entry.
      TAG_TEMPLATE: A TagTemplate.
      ENTRY_GROUP: An EntryGroup.
    """
        SEARCH_RESULT_TYPE_UNSPECIFIED = 0
        ENTRY = 1
        TAG_TEMPLATE = 2
        ENTRY_GROUP = 3
    description = _messages.StringField(1)
    displayName = _messages.StringField(2)
    fullyQualifiedName = _messages.StringField(3)
    integratedSystem = _messages.EnumField('IntegratedSystemValueValuesEnum', 4)
    linkedResource = _messages.StringField(5)
    modifyTime = _messages.StringField(6)
    relativeResourceName = _messages.StringField(7)
    searchResultSubtype = _messages.StringField(8)
    searchResultType = _messages.EnumField('SearchResultTypeValueValuesEnum', 9)
    userSpecifiedSystem = _messages.StringField(10)