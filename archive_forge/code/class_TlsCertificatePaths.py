from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TlsCertificatePaths(_messages.Message):
    """[Deprecated] The paths to the mounted TLS Certificates and private key.
  The paths to the mounted TLS Certificates and private key.

  Fields:
    certificatePath: The path to the file holding the client or server TLS
      certificate to use.
    privateKeyPath: The path to the file holding the client or server private
      key.
  """
    certificatePath = _messages.StringField(1)
    privateKeyPath = _messages.StringField(2)