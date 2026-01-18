from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApplyHash(_messages.Message):
    """Apply a hash function on the value.

  Fields:
    uuidFromBytes: Optional. Generate UUID from the data's byte array
  """
    uuidFromBytes = _messages.MessageField('Empty', 1)