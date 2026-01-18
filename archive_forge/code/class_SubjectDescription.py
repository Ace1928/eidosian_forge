from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SubjectDescription(_messages.Message):
    """These values describe fields in an issued X.509 certificate such as the
  distinguished name, subject alternative names, serial number, and lifetime.

  Fields:
    hexSerialNumber: The serial number encoded in lowercase hexadecimal.
    lifetime: For convenience, the actual lifetime of an issued certificate.
    notAfterTime: The time after which the certificate is expired. Per RFC
      5280, the validity period for a certificate is the period of time from
      not_before_time through not_after_time, inclusive. Corresponds to
      'not_before_time' + 'lifetime' - 1 second.
    notBeforeTime: The time at which the certificate becomes valid.
    subject: Contains distinguished name fields such as the common name,
      location and / organization.
    subjectAltName: The subject alternative name fields.
  """
    hexSerialNumber = _messages.StringField(1)
    lifetime = _messages.StringField(2)
    notAfterTime = _messages.StringField(3)
    notBeforeTime = _messages.StringField(4)
    subject = _messages.MessageField('Subject', 5)
    subjectAltName = _messages.MessageField('SubjectAltNames', 6)