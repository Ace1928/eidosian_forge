from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GroupState(_messages.Message):
    """Information about the state of a group with respect to a consumer
  resource.

  Fields:
    group: Output only. The group referenced by this state.
    name: Output only. The resource name of the group state.
    state: Output only. The state of this group with respect to the consumer
      parent.
  """
    group = _messages.MessageField('GoogleApiServiceusageV2alphaGroup', 1)
    name = _messages.StringField(2)
    state = _messages.MessageField('State', 3)