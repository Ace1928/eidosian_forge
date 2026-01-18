from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SubjectAltNames(_messages.Message):
    """SubjectAltNames corresponds to a more modern way of listing what the
  asserted identity is in a certificate (i.e., compared to the "common name"
  in the distinguished name).

  Fields:
    customSans: Contains additional subject alternative name values. For each
      custom_san, the `value` field must contain an ASN.1 encoded UTF8String.
    dnsNames: Contains only valid, fully-qualified host names.
    emailAddresses: Contains only valid RFC 2822 E-mail addresses.
    ipAddresses: Contains only valid 32-bit IPv4 addresses or RFC 4291 IPv6
      addresses.
    uris: Contains only valid RFC 3986 URIs.
  """
    customSans = _messages.MessageField('X509Extension', 1, repeated=True)
    dnsNames = _messages.StringField(2, repeated=True)
    emailAddresses = _messages.StringField(3, repeated=True)
    ipAddresses = _messages.StringField(4, repeated=True)
    uris = _messages.StringField(5, repeated=True)