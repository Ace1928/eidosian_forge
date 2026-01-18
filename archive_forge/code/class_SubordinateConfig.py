from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SubordinateConfig(_messages.Message):
    """Describes a subordinate CA's issuers. This is either a resource name to
  a known issuing CertificateAuthority, or a PEM issuer certificate chain.

  Fields:
    certificateAuthority: Required. This can refer to a CertificateAuthority
      that was used to create a subordinate CertificateAuthority. This field
      is used for information and usability purposes only. The resource name
      is in the format
      `projects/*/locations/*/caPools/*/certificateAuthorities/*`.
    pemIssuerChain: Required. Contains the PEM certificate chain for the
      issuers of this CertificateAuthority, but not pem certificate for this
      CA itself.
  """
    certificateAuthority = _messages.StringField(1)
    pemIssuerChain = _messages.MessageField('SubordinateConfigChain', 2)