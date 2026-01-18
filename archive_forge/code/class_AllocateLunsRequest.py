from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AllocateLunsRequest(_messages.Message):
    """Message for creating Luns for Volume.

  Fields:
    lunRanges: Required. LUN ranges to be allocated.
  """
    lunRanges = _messages.MessageField('AllocateLunsRequestLunRange', 1, repeated=True)