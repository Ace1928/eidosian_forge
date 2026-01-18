from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SslCertificateSelfManagedSslCertificate(_messages.Message):
    """Configuration and status of a self-managed SSL certificate.

  Fields:
    certificate: A local certificate file. The certificate must be in PEM
      format. The certificate chain must be no greater than 5 certs long. The
      chain must include at least one intermediate cert.
    privateKey: A write-only private key in PEM format. Only insert requests
      will include this field.
  """
    certificate = _messages.StringField(1)
    privateKey = _messages.StringField(2)