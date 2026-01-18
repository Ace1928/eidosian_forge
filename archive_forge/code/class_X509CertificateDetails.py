from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class X509CertificateDetails(_messages.Message):
    """Details of an X.509 certificate. For informational purposes only.

  Fields:
    expiryTime: The time the certificate becomes invalid.
    issuer: The entity that signed the certificate.
    publicKeyType: The type of public key in the certificate.
    signatureAlgorithm: The algorithm used to sign the certificate.
    startTime: The time the certificate becomes valid.
    subject: The entity the certificate and public key belong to.
  """
    expiryTime = _messages.StringField(1)
    issuer = _messages.StringField(2)
    publicKeyType = _messages.StringField(3)
    signatureAlgorithm = _messages.StringField(4)
    startTime = _messages.StringField(5)
    subject = _messages.StringField(6)