from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OracleConnectionProfile(_messages.Message):
    """Specifies connection parameters required specifically for Oracle
  databases.

  Fields:
    databaseService: Required. Database service for the Oracle connection.
    forwardSshConnectivity: Forward SSH tunnel connectivity.
    host: Required. The IP or hostname of the source Oracle database.
    password: Required. Input only. The password for the user that Database
      Migration Service will be using to connect to the database. This field
      is not returned on request, and the value is encrypted when stored in
      Database Migration Service.
    passwordSet: Output only. Indicates whether a new password is included in
      the request.
    port: Required. The network port of the source Oracle database.
    privateConnectivity: Private connectivity.
    ssl: SSL configuration for the connection to the source Oracle database. *
      Only `SERVER_ONLY` configuration is supported for Oracle SSL. * SSL is
      supported for Oracle versions 12 and above.
    staticServiceIpConnectivity: Static Service IP connectivity.
    username: Required. The username that Database Migration Service will use
      to connect to the database. The value is encrypted when stored in
      Database Migration Service.
  """
    databaseService = _messages.StringField(1)
    forwardSshConnectivity = _messages.MessageField('ForwardSshTunnelConnectivity', 2)
    host = _messages.StringField(3)
    password = _messages.StringField(4)
    passwordSet = _messages.BooleanField(5)
    port = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    privateConnectivity = _messages.MessageField('PrivateConnectivity', 7)
    ssl = _messages.MessageField('SslConfig', 8)
    staticServiceIpConnectivity = _messages.MessageField('StaticServiceIpConnectivity', 9)
    username = _messages.StringField(10)