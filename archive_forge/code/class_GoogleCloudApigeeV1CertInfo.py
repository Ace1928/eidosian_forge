from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1CertInfo(_messages.Message):
    """X.509 certificate as defined in RFC 5280.

  Fields:
    basicConstraints: X.509 basic constraints extension.
    expiryDate: X.509 `notAfter` validity period in milliseconds since epoch.
    isValid: Flag that specifies whether the certificate is valid. Flag is set
      to `Yes` if the certificate is valid, `No` if expired, or `Not yet` if
      not yet valid.
    issuer: X.509 issuer.
    publicKey: Public key component of the X.509 subject public key info.
    serialNumber: X.509 serial number.
    sigAlgName: X.509 signatureAlgorithm.
    subject: X.509 subject.
    subjectAlternativeNames: X.509 subject alternative names (SANs) extension.
    validFrom: X.509 `notBefore` validity period in milliseconds since epoch.
    version: X.509 version.
  """
    basicConstraints = _messages.StringField(1)
    expiryDate = _messages.IntegerField(2)
    isValid = _messages.StringField(3)
    issuer = _messages.StringField(4)
    publicKey = _messages.StringField(5)
    serialNumber = _messages.StringField(6)
    sigAlgName = _messages.StringField(7)
    subject = _messages.StringField(8)
    subjectAlternativeNames = _messages.StringField(9, repeated=True)
    validFrom = _messages.IntegerField(10)
    version = _messages.IntegerField(11, variant=_messages.Variant.INT32)