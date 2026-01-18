from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceGroupManagerStatusStatefulPerInstanceConfigs(_messages.Message):
    """A InstanceGroupManagerStatusStatefulPerInstanceConfigs object.

  Fields:
    allEffective: A bit indicating if all of the group's per-instance
      configurations (listed in the output of a listPerInstanceConfigs API
      call) have status EFFECTIVE or there are no per-instance-configs.
  """
    allEffective = _messages.BooleanField(1)