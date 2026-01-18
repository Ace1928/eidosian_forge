from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SubjectConfig(_messages.Message):
    """These values are used to create the distinguished name and subject
  alternative name fields in an X.509 certificate.

  Fields:
    subject: Optional. Contains distinguished name fields such as the common
      name, location and organization.
    subjectAltName: Optional. The subject alternative name fields.
  """
    subject = _messages.MessageField('Subject', 1)
    subjectAltName = _messages.MessageField('SubjectAltNames', 2)