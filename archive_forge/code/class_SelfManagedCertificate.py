from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SelfManagedCertificate(_messages.Message):
    """Certificate data for a SelfManaged Certificate. SelfManaged Certificates
  are uploaded by the user. Updating such certificates before they expire
  remains the user's responsibility.

  Fields:
    certificatePem: Input only. The certificate chain in PEM-encoded form.
      Leaf certificate comes first, followed by intermediate ones if any.
    privateKeyPem: Input only. The private key of the leaf certificate in PEM-
      encoded form.
  """
    certificatePem = _messages.StringField(1)
    privateKeyPem = _messages.StringField(2)