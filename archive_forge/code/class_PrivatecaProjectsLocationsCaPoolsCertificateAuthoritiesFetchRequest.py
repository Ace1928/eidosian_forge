from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesFetchRequest(_messages.Message):
    """A PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesFetchRequest
  object.

  Fields:
    name: Required. The resource name for this CertificateAuthority in the
      format `projects/*/locations/*/caPools/*/certificateAuthorities/*`.
  """
    name = _messages.StringField(1, required=True)