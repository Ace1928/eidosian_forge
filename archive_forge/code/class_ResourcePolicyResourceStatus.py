from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourcePolicyResourceStatus(_messages.Message):
    """Contains output only fields. Use this sub-message for all output fields
  set on ResourcePolicy. The internal structure of this "status" field should
  mimic the structure of ResourcePolicy proto specification.

  Fields:
    instanceSchedulePolicy: [Output Only] Specifies a set of output values
      reffering to the instance_schedule_policy system status. This field
      should have the same name as corresponding policy field.
  """
    instanceSchedulePolicy = _messages.MessageField('ResourcePolicyResourceStatusInstanceSchedulePolicyStatus', 1)