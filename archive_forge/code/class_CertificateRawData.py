from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CertificateRawData(_messages.Message):
    """An SSL certificate obtained from a certificate authority.

  Fields:
    privateKey: Unencrypted PEM encoded RSA private key. This field is set
      once on certificate creation and then encrypted. The key size must be
      2048 bits or fewer. Must include the header and footer. Example:
      -----BEGIN RSA PRIVATE KEY----- -----END RSA PRIVATE KEY----- @InputOnly
    publicCertificate: PEM encoded x.509 public key certificate. This field is
      set once on certificate creation. Must include the header and footer.
      Example: -----BEGIN CERTIFICATE----- -----END CERTIFICATE-----
  """
    privateKey = _messages.StringField(1)
    publicCertificate = _messages.StringField(2)