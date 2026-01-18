from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SqlServerConnectionProfile(_messages.Message):
    """Specifies connection parameters required specifically for SQL Server
  databases.

  Fields:
    backups: The backup details in Cloud Storage for homogeneous migration to
      Cloud SQL for SQL Server.
    cloudSqlId: If the source is a Cloud SQL database, use this field to
      provide the Cloud SQL instance ID of the source.
    forwardSshConnectivity: Forward SSH tunnel connectivity.
    host: Required. The IP or hostname of the source SQL Server database.
    password: Required. Input only. The password for the user that Database
      Migration Service will be using to connect to the database. This field
      is not returned on request, and the value is encrypted when stored in
      Database Migration Service.
    passwordSet: Output only. Indicates whether a new password is included in
      the request.
    port: Required. The network port of the source SQL Server database.
    privateConnectivity: Private connectivity.
    privateServiceConnectConnectivity: Private Service Connect connectivity.
    ssl: SSL configuration for the destination to connect to the source
      database.
    staticIpConnectivity: Static IP connectivity data (default, no additional
      details needed).
    username: Required. The username that Database Migration Service will use
      to connect to the database. The value is encrypted when stored in
      Database Migration Service.
  """
    backups = _messages.MessageField('SqlServerBackups', 1)
    cloudSqlId = _messages.StringField(2)
    forwardSshConnectivity = _messages.MessageField('ForwardSshTunnelConnectivity', 3)
    host = _messages.StringField(4)
    password = _messages.StringField(5)
    passwordSet = _messages.BooleanField(6)
    port = _messages.IntegerField(7, variant=_messages.Variant.INT32)
    privateConnectivity = _messages.MessageField('PrivateConnectivity', 8)
    privateServiceConnectConnectivity = _messages.MessageField('PrivateServiceConnectConnectivity', 9)
    ssl = _messages.MessageField('SslConfig', 10)
    staticIpConnectivity = _messages.MessageField('StaticIpConnectivity', 11)
    username = _messages.StringField(12)