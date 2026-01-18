from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SelfManagedCertData(_messages.Message):
    """Uploaded certificate data.

  Fields:
    certificatePem: The certificate chain in PEM-encoded form.
    privateKeyPem: Input only. The private key data in PEM-encoded form.
  """
    certificatePem = _messages.BytesField(1)
    privateKeyPem = _messages.BytesField(2)