from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1SqlDatabaseSystemSpec(_messages.Message):
    """Specification that applies to entries that are part `SQL_DATABASE`
  system (user_specified_type)

  Fields:
    databaseVersion: Version of the database engine.
    instanceHost: Host of the SQL database enum InstanceHost { UNDEFINED = 0;
      SELF_HOSTED = 1; CLOUD_SQL = 2; AMAZON_RDS = 3; AZURE_SQL = 4; } Host of
      the enclousing database instance.
    sqlEngine: SQL Database Engine. enum SqlEngine { UNDEFINED = 0; MY_SQL =
      1; POSTGRE_SQL = 2; SQL_SERVER = 3; } Engine of the enclosing database
      instance.
  """
    databaseVersion = _messages.StringField(1)
    instanceHost = _messages.StringField(2)
    sqlEngine = _messages.StringField(3)