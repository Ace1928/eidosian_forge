from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TlsCertificateFiles(_messages.Message):
    """Specification of TLS certificate files.

  Fields:
    certificatePath: Required. The path to the file that has the certificate
      containing public key.
    privateKeyPath: Required. The path to the file that has the private key.
  """
    certificatePath = _messages.StringField(1)
    privateKeyPath = _messages.StringField(2)