from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1CloudSqlBigQueryConnectionSpec(_messages.Message):
    """Specification for the BigQuery connection to a Cloud SQL instance.

  Enums:
    TypeValueValuesEnum: Type of the Cloud SQL database.

  Fields:
    database: Database name.
    instanceId: Cloud SQL instance ID in the format of
      `project:location:instance`.
    type: Type of the Cloud SQL database.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Type of the Cloud SQL database.

    Values:
      DATABASE_TYPE_UNSPECIFIED: Unspecified database type.
      POSTGRES: Cloud SQL for PostgreSQL.
      MYSQL: Cloud SQL for MySQL.
    """
        DATABASE_TYPE_UNSPECIFIED = 0
        POSTGRES = 1
        MYSQL = 2
    database = _messages.StringField(1)
    instanceId = _messages.StringField(2)
    type = _messages.EnumField('TypeValueValuesEnum', 3)