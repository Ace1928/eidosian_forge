from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SchemaSegment(_messages.Message):
    """An HL7v2 Segment.

  Fields:
    maxOccurs: The maximum number of times this segment can be present in this
      group. 0 or -1 means unbounded.
    minOccurs: The minimum number of times this segment can be present in this
      group.
    type: The Segment type. For example, "PID".
  """
    maxOccurs = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    minOccurs = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    type = _messages.StringField(3)