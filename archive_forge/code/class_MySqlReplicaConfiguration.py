from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class MySqlReplicaConfiguration(_messages.Message):
    """Read-replica configuration specific to MySQL databases.

  Fields:
    caCertificate: PEM representation of the trusted CA's x509 certificate.
    clientCertificate: PEM representation of the replica's x509 certificate.
    clientKey: PEM representation of the replica's private key. The
      corresponsing public key is encoded in the client's certificate.
    connectRetryInterval: Seconds to wait between connect retries. MySQL's
      default is 60 seconds.
    dumpFilePath: Path to a SQL dump file in Google Cloud Storage from which
      the replica instance is to be created. The URI is in the form
      gs://bucketName/fileName. Compressed gzip files (.gz) are also
      supported. Dumps have the binlog co-ordinates from which replication
      begins. This can be accomplished by setting --master-data to 1 when
      using mysqldump.
    kind: This is always `sql#mysqlReplicaConfiguration`.
    masterHeartbeatPeriod: Interval in milliseconds between replication
      heartbeats.
    password: The password for the replication connection.
    sslCipher: A list of permissible ciphers to use for SSL encryption.
    username: The username for the replication connection.
    verifyServerCertificate: Whether or not to check the primary instance's
      Common Name value in the certificate that it sends during the SSL
      handshake.
  """
    caCertificate = _messages.StringField(1)
    clientCertificate = _messages.StringField(2)
    clientKey = _messages.StringField(3)
    connectRetryInterval = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    dumpFilePath = _messages.StringField(5)
    kind = _messages.StringField(6)
    masterHeartbeatPeriod = _messages.IntegerField(7)
    password = _messages.StringField(8)
    sslCipher = _messages.StringField(9)
    username = _messages.StringField(10)
    verifyServerCertificate = _messages.BooleanField(11)