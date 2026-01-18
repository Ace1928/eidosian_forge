from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class DemoteMasterMySqlReplicaConfiguration(_messages.Message):
    """Read-replica configuration specific to MySQL databases.

  Fields:
    caCertificate: PEM representation of the trusted CA's x509 certificate.
    clientCertificate: PEM representation of the replica's x509 certificate.
    clientKey: PEM representation of the replica's private key. The
      corresponsing public key is encoded in the client's certificate. The
      format of the replica's private key can be either PKCS #1 or PKCS #8.
    kind: This is always `sql#demoteMasterMysqlReplicaConfiguration`.
    password: The password for the replication connection.
    username: The username for the replication connection.
  """
    caCertificate = _messages.StringField(1)
    clientCertificate = _messages.StringField(2)
    clientKey = _messages.StringField(3)
    kind = _messages.StringField(4)
    password = _messages.StringField(5)
    username = _messages.StringField(6)