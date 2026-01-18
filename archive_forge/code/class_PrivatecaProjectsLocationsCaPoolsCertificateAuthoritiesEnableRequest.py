from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesEnableRequest(_messages.Message):
    """A PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesEnableRequest
  object.

  Fields:
    enableCertificateAuthorityRequest: A EnableCertificateAuthorityRequest
      resource to be passed as the request body.
    name: Required. The resource name for this CertificateAuthority in the
      format `projects/*/locations/*/caPools/*/certificateAuthorities/*`.
  """
    enableCertificateAuthorityRequest = _messages.MessageField('EnableCertificateAuthorityRequest', 1)
    name = _messages.StringField(2, required=True)