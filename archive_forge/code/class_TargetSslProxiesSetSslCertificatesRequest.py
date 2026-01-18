from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TargetSslProxiesSetSslCertificatesRequest(_messages.Message):
    """A TargetSslProxiesSetSslCertificatesRequest object.

  Fields:
    sslCertificates: New set of URLs to SslCertificate resources to associate
      with this TargetSslProxy. At least one SSL certificate must be
      specified. Currently, you may specify up to 15 SSL certificates.
  """
    sslCertificates = _messages.StringField(1, repeated=True)