from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DetachTrustRequest(_messages.Message):
    """Request message for DetachTrust

  Fields:
    trust: Required. The domain trust resource to removed.
  """
    trust = _messages.MessageField('Trust', 1)