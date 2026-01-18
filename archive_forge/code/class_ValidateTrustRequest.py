from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ValidateTrustRequest(_messages.Message):
    """Request message for ValidateTrust

  Fields:
    trust: Required. The domain trust to validate trust state for.
  """
    trust = _messages.MessageField('Trust', 1)