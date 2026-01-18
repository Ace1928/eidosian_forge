from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeTimeCursorResponse(_messages.Message):
    """Response containing the cursor corresponding to a publish or event time
  in a topic partition.

  Fields:
    cursor: If present, the cursor references the first message with time
      greater than or equal to the specified target time. If such a message
      cannot be found, the cursor will be unset (i.e. `cursor` is not
      present).
  """
    cursor = _messages.MessageField('Cursor', 1)