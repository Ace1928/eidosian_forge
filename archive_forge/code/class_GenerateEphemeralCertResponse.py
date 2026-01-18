from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GenerateEphemeralCertResponse(_messages.Message):
    """Ephemeral certificate creation request.

  Fields:
    ephemeralCert: Generated cert
  """
    ephemeralCert = _messages.MessageField('SslCert', 1)