from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Encryption(_messages.Message):
    """Encryption message describes the details of the applied encryption.

  Fields:
    kmsKey: Required. The name of the encryption key that is stored in Google
      Cloud KMS.
  """
    kmsKey = _messages.StringField(1)