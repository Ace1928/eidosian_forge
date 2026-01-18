from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Subject(_messages.Message):
    """Subject describes parts of a distinguished name that, in turn, describes
  the subject of the certificate.

  Fields:
    commonName: The "common name" of the subject.
    countryCode: The country code of the subject.
    locality: The locality or city of the subject.
    organization: The organization of the subject.
    organizationalUnit: The organizational_unit of the subject.
    postalCode: The postal code of the subject.
    province: The province, territory, or regional state of the subject.
    streetAddress: The street address of the subject.
  """
    commonName = _messages.StringField(1)
    countryCode = _messages.StringField(2)
    locality = _messages.StringField(3)
    organization = _messages.StringField(4)
    organizationalUnit = _messages.StringField(5)
    postalCode = _messages.StringField(6)
    province = _messages.StringField(7)
    streetAddress = _messages.StringField(8)