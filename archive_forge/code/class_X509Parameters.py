from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class X509Parameters(_messages.Message):
    """An X509Parameters is used to describe certain fields of an X.509
  certificate, such as the key usage fields, fields specific to CA
  certificates, certificate policy extensions and custom extensions.

  Fields:
    additionalExtensions: Optional. Describes custom X.509 extensions.
    aiaOcspServers: Optional. Describes Online Certificate Status Protocol
      (OCSP) endpoint addresses that appear in the "Authority Information
      Access" extension in the certificate.
    caOptions: Optional. Describes options in this X509Parameters that are
      relevant in a CA certificate.
    keyUsage: Optional. Indicates the intended use for keys that correspond to
      a certificate.
    nameConstraints: Optional. Describes the X.509 name constraints extension.
    policyIds: Optional. Describes the X.509 certificate policy object
      identifiers, per https://tools.ietf.org/html/rfc5280#section-4.2.1.4.
  """
    additionalExtensions = _messages.MessageField('X509Extension', 1, repeated=True)
    aiaOcspServers = _messages.StringField(2, repeated=True)
    caOptions = _messages.MessageField('CaOptions', 3)
    keyUsage = _messages.MessageField('KeyUsage', 4)
    nameConstraints = _messages.MessageField('NameConstraints', 5)
    policyIds = _messages.MessageField('ObjectId', 6, repeated=True)