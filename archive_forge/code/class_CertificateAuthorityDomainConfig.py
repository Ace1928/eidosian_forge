from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CertificateAuthorityDomainConfig(_messages.Message):
    """CertificateAuthorityDomainConfig configures one or more fully qualified
  domain names (FQDN) to a specific certificate.

  Fields:
    fqdns: List of fully qualified domain names (FQDN). Specifying port is
      supported. Wilcards are NOT supported. Examples: - my.customdomain.com -
      10.0.1.2:5000
    gcpSecretManagerCertificateConfig: Google Secret Manager (GCP) certificate
      configuration.
  """
    fqdns = _messages.StringField(1, repeated=True)
    gcpSecretManagerCertificateConfig = _messages.MessageField('GCPSecretManagerCertificateConfig', 2)