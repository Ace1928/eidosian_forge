from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2CloudSqlProperties(_messages.Message):
    """Cloud SQL connection properties.

  Enums:
    DatabaseEngineValueValuesEnum: Required. The database engine used by the
      Cloud SQL instance that this connection configures.

  Fields:
    cloudSqlIam: Built-in IAM authentication (must be configured in Cloud
      SQL).
    connectionName: Optional. Immutable. The Cloud SQL instance for which the
      connection is defined. Only one connection per instance is allowed. This
      can only be set at creation time, and cannot be updated. It is an error
      to use a connection_name from different project or region than the one
      that holds the connection. For example, a Connection resource for Cloud
      SQL connection_name "project-id:us-central1:sql-instance" must be
      created under the parent "projects/project-id/locations/us-central1"
    databaseEngine: Required. The database engine used by the Cloud SQL
      instance that this connection configures.
    maxConnections: Required. DLP will limit its connections to
      max_connections. Must be 2 or greater.
    usernamePassword: A username and password stored in Secret Manager.
  """

    class DatabaseEngineValueValuesEnum(_messages.Enum):
        """Required. The database engine used by the Cloud SQL instance that this
    connection configures.

    Values:
      DATABASE_ENGINE_UNKNOWN: An engine that is not currently supported by
        SDP.
      DATABASE_ENGINE_MYSQL: Cloud SQL for MySQL instance.
      DATABASE_ENGINE_POSTGRES: Cloud SQL for Postgres instance.
    """
        DATABASE_ENGINE_UNKNOWN = 0
        DATABASE_ENGINE_MYSQL = 1
        DATABASE_ENGINE_POSTGRES = 2
    cloudSqlIam = _messages.MessageField('GooglePrivacyDlpV2CloudSqlIamCredential', 1)
    connectionName = _messages.StringField(2)
    databaseEngine = _messages.EnumField('DatabaseEngineValueValuesEnum', 3)
    maxConnections = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    usernamePassword = _messages.MessageField('GooglePrivacyDlpV2SecretManagerCredential', 5)