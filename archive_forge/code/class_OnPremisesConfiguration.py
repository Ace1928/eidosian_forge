from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OnPremisesConfiguration(_messages.Message):
    """On-premises instance configuration.

  Fields:
    caCertificate: PEM representation of the trusted CA's x509 certificate.
    clientCertificate: PEM representation of the replica's x509 certificate.
    clientKey: PEM representation of the replica's private key. The
      corresponsing public key is encoded in the client's certificate.
    dumpFilePath: The dump file to create the Cloud SQL replica.
    hostPort: The host and port of the on-premises instance in host:port
      format
    kind: This is always `sql#onPremisesConfiguration`.
    password: The password for connecting to on-premises instance.
    sourceInstance: The reference to Cloud SQL instance if the source is Cloud
      SQL.
    username: The username for connecting to on-premises instance.
  """
    caCertificate = _messages.StringField(1)
    clientCertificate = _messages.StringField(2)
    clientKey = _messages.StringField(3)
    dumpFilePath = _messages.StringField(4)
    hostPort = _messages.StringField(5)
    kind = _messages.StringField(6)
    password = _messages.StringField(7)
    sourceInstance = _messages.MessageField('InstanceReference', 8)
    username = _messages.StringField(9)