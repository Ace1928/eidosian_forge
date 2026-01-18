from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AttachTrustRequest(_messages.Message):
    """Request message for AttachTrust

  Fields:
    trust: Required. The domain trust resource.
  """
    trust = _messages.MessageField('Trust', 1)