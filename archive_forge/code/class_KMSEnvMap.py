from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KMSEnvMap(_messages.Message):
    """A KMSEnvMap object.

  Fields:
    cipherText: The value of the cipherText response from the `encrypt`
      method.
    keyName: The name of the KMS key that will be used to decrypt the cipher
      text.
  """
    cipherText = _messages.StringField(1)
    keyName = _messages.StringField(2)