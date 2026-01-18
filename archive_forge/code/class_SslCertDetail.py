from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SslCertDetail(_messages.Message):
    """SslCertDetail.

  Fields:
    certInfo: The public information about the cert.
    certPrivateKey: The private key for the client cert, in pem format. Keep
      private in order to protect your security.
  """
    certInfo = _messages.MessageField('SslCert', 1)
    certPrivateKey = _messages.StringField(2)