from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RegistryCredential(_messages.Message):
    """A server-stored registry credential used to validate device credentials.

  Fields:
    publicKeyCertificate: A public key certificate used to verify the device
      credentials.
  """
    publicKeyCertificate = _messages.MessageField('PublicKeyCertificate', 1)