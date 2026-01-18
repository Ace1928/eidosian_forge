from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageDatabasecenterProtoCommonProduct(_messages.Message):
    """Product specification for Condor resources.

  Enums:
    EngineValueValuesEnum: The specific engine that the underlying database is
      running.
    TypeValueValuesEnum: Type of specific database product. It could be
      CloudSQL, AlloyDB etc..

  Fields:
    engine: The specific engine that the underlying database is running.
    type: Type of specific database product. It could be CloudSQL, AlloyDB
      etc..
    version: Version of the underlying database engine. Example values: For
      MySQL, it could be "8.0", "5.7" etc.. For Postgres, it could be "14",
      "15" etc..
  """

    class EngineValueValuesEnum(_messages.Enum):
        """The specific engine that the underlying database is running.

    Values:
      ENGINE_UNSPECIFIED: UNSPECIFIED means engine type is not known or
        available.
      ENGINE_MYSQL: MySQL binary running as an engine in the database
        instance.
      MYSQL: MySQL binary running as engine in database instance.
      ENGINE_POSTGRES: Postgres binary running as engine in database instance.
      POSTGRES: Postgres binary running as engine in database instance.
      ENGINE_SQL_SERVER: SQLServer binary running as engine in database
        instance.
      SQL_SERVER: SQLServer binary running as engine in database instance.
      ENGINE_NATIVE: Native database binary running as engine in instance.
      NATIVE: Native database binary running as engine in instance.
      ENGINE_CLOUD_SPANNER_WITH_POSTGRES_DIALECT: Cloud Spanner with
        PostgreSQL dialect.
      ENGINE_CLOUD_SPANNER_WITH_GOOGLESQL_DIALECT: Cloud Spanner with Google
        SQL dialect.
      ENGINE_MEMORYSTORE_FOR_REDIS: Memorystore with Redis dialect.
      ENGINE_MEMORYSTORE_FOR_REDIS_CLUSTER: Memorystore with Redis cluster
        dialect.
      ENGINE_OTHER: Other refers to rest of other database engine. This is to
        be when engine is known, but it is not present in this enum.
    """
        ENGINE_UNSPECIFIED = 0
        ENGINE_MYSQL = 1
        MYSQL = 2
        ENGINE_POSTGRES = 3
        POSTGRES = 4
        ENGINE_SQL_SERVER = 5
        SQL_SERVER = 6
        ENGINE_NATIVE = 7
        NATIVE = 8
        ENGINE_CLOUD_SPANNER_WITH_POSTGRES_DIALECT = 9
        ENGINE_CLOUD_SPANNER_WITH_GOOGLESQL_DIALECT = 10
        ENGINE_MEMORYSTORE_FOR_REDIS = 11
        ENGINE_MEMORYSTORE_FOR_REDIS_CLUSTER = 12
        ENGINE_OTHER = 13

    class TypeValueValuesEnum(_messages.Enum):
        """Type of specific database product. It could be CloudSQL, AlloyDB etc..

    Values:
      PRODUCT_TYPE_UNSPECIFIED: UNSPECIFIED means product type is not known or
        available.
      PRODUCT_TYPE_CLOUD_SQL: Cloud SQL product area in GCP
      CLOUD_SQL: Cloud SQL product area in GCP
      PRODUCT_TYPE_ALLOYDB: AlloyDB product area in GCP
      ALLOYDB: AlloyDB product area in GCP
      PRODUCT_TYPE_SPANNER: Spanner product area in GCP
      PRODUCT_TYPE_ON_PREM: On premises database product.
      ON_PREM: On premises database product.
      PRODUCT_TYPE_MEMORYSTORE: Memorystore product area in GCP
      PRODUCT_TYPE_OTHER: Other refers to rest of other product type. This is
        to be when product type is known, but it is not present in this enum.
    """
        PRODUCT_TYPE_UNSPECIFIED = 0
        PRODUCT_TYPE_CLOUD_SQL = 1
        CLOUD_SQL = 2
        PRODUCT_TYPE_ALLOYDB = 3
        ALLOYDB = 4
        PRODUCT_TYPE_SPANNER = 5
        PRODUCT_TYPE_ON_PREM = 6
        ON_PREM = 7
        PRODUCT_TYPE_MEMORYSTORE = 8
        PRODUCT_TYPE_OTHER = 9
    engine = _messages.EnumField('EngineValueValuesEnum', 1)
    type = _messages.EnumField('TypeValueValuesEnum', 2)
    version = _messages.StringField(3)