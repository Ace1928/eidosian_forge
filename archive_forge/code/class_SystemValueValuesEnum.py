from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SystemValueValuesEnum(_messages.Enum):
    """Service in which the external table is registered.

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