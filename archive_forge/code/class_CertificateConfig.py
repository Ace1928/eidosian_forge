from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CertificateConfig(_messages.Message):
    """A CertificateConfig describes an X.509 certificate or CSR that is to be
  created, as an alternative to using ASN.1.

  Fields:
    publicKey: Optional. The public key that corresponds to this config. This
      is, for example, used when issuing Certificates, but not when creating a
      self-signed CertificateAuthority or CertificateAuthority CSR.
    subjectConfig: Required. Specifies some of the values in a certificate
      that are related to the subject.
    subjectKeyId: Optional. When specified this provides a custom SKI to be
      used in the certificate. This should only be used to maintain a SKI of
      an existing CA originally created outside CA service, which was not
      generated using method (1) described in RFC 5280 section 4.2.1.2.
    x509Config: Required. Describes how some of the technical X.509 fields in
      a certificate should be populated.
  """
    publicKey = _messages.MessageField('PublicKey', 1)
    subjectConfig = _messages.MessageField('SubjectConfig', 2)
    subjectKeyId = _messages.MessageField('CertificateConfigKeyId', 3)
    x509Config = _messages.MessageField('X509Parameters', 4)