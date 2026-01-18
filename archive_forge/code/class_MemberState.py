from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MemberState(_messages.Message):
    """Information about the member state with respect to a particular
  consumer.

  Fields:
    member: Output only. The member referenced by this state.
    name: Output only. The resource name of the member state.
  """
    member = _messages.MessageField('Member', 1)
    name = _messages.StringField(2)