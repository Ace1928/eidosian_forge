from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SchemaGroup(_messages.Message):
    """An HL7v2 logical group construct.

  Fields:
    choice: True indicates that this is a choice group, meaning that only one
      of its segments can exist in a given message.
    maxOccurs: The maximum number of times this group can be repeated. 0 or -1
      means unbounded.
    members: Nested groups and/or segments.
    minOccurs: The minimum number of times this group must be
      present/repeated.
    name: The name of this group. For example, "ORDER_DETAIL".
  """
    choice = _messages.BooleanField(1)
    maxOccurs = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    members = _messages.MessageField('GroupOrSegment', 3, repeated=True)
    minOccurs = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    name = _messages.StringField(5)