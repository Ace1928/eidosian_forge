from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2UnwrappedCryptoKey(_messages.Message):
    """Using raw keys is prone to security risks due to accidentally leaking
  the key. Choose another type of key if possible.

  Fields:
    key: Required. A 128/192/256 bit key.
  """
    key = _messages.BytesField(1)