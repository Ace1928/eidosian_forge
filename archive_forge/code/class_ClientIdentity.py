from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClientIdentity(_messages.Message):
    """Client Identity for TLS authentication with the Web3Signer service. This
  has the common name of the client and the SHA-256 fingerprint for a self-
  signed X.509 certificate. Reference:
  https://docs.web3signer.consensys.io/how-to/configure-tls#create-the-known-
  clients-file

  Fields:
    certificateCn: Output only. Common Name (CN) for the lighthouse client
    clientCertificateFingerprint: Output only. SHA-256 fingerprint of the
      client's self-signed certificate
  """
    certificateCn = _messages.StringField(1)
    clientCertificateFingerprint = _messages.StringField(2)