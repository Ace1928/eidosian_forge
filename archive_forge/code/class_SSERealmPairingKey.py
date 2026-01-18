from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SSERealmPairingKey(_messages.Message):
    """Key to be shared with SSE service provider to establish global handshake

  Fields:
    expireTime: Output only. Timestamp in UTC of when this resource is
      considered expired.
    key: Output only. The name of the key. It expires 7 days after creation.
  """
    expireTime = _messages.StringField(1)
    key = _messages.StringField(2)