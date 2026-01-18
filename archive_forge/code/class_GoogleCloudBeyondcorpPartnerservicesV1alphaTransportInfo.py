from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBeyondcorpPartnerservicesV1alphaTransportInfo(_messages.Message):
    """Message contains the transport layer information to verify the proxy
  server.

  Fields:
    serverCaCertPem: Required. PEM encoded CA certificate associated with the
      proxy server certificate.
    sslDecryptCaCertPem: Optional. PEM encoded CA certificate associated with
      the certificate used by proxy server for SSL decryption.
  """
    serverCaCertPem = _messages.StringField(1)
    sslDecryptCaCertPem = _messages.StringField(2)