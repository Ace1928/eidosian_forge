from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SourceConfig(_messages.Message):
    """The configuration of the stream source.

  Fields:
    mysqlSourceConfig: MySQL data source configuration.
    oracleSourceConfig: Oracle data source configuration.
    postgresqlSourceConfig: PostgreSQL data source configuration.
    sourceConnectionProfile: Required. Source connection profile resoource.
      Format:
      `projects/{project}/locations/{location}/connectionProfiles/{name}`
    sqlServerSourceConfig: SQLServer data source configuration.
  """
    mysqlSourceConfig = _messages.MessageField('MysqlSourceConfig', 1)
    oracleSourceConfig = _messages.MessageField('OracleSourceConfig', 2)
    postgresqlSourceConfig = _messages.MessageField('PostgresqlSourceConfig', 3)
    sourceConnectionProfile = _messages.StringField(4)
    sqlServerSourceConfig = _messages.MessageField('SqlServerSourceConfig', 5)