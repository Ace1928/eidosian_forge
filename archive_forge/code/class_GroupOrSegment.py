from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GroupOrSegment(_messages.Message):
    """Construct representing a logical group or a segment.

  Fields:
    group: A SchemaGroup attribute.
    segment: A SchemaSegment attribute.
  """
    group = _messages.MessageField('SchemaGroup', 1)
    segment = _messages.MessageField('SchemaSegment', 2)