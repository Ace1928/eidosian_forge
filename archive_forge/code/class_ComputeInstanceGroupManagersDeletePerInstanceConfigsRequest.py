from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeInstanceGroupManagersDeletePerInstanceConfigsRequest(_messages.Message):
    """A ComputeInstanceGroupManagersDeletePerInstanceConfigsRequest object.

  Fields:
    instanceGroupManager: The name of the managed instance group. It should
      conform to RFC1035.
    instanceGroupManagersDeletePerInstanceConfigsReq: A
      InstanceGroupManagersDeletePerInstanceConfigsReq resource to be passed
      as the request body.
    project: Project ID for this request.
    zone: The name of the zone where the managed instance group is located. It
      should conform to RFC1035.
  """
    instanceGroupManager = _messages.StringField(1, required=True)
    instanceGroupManagersDeletePerInstanceConfigsReq = _messages.MessageField('InstanceGroupManagersDeletePerInstanceConfigsReq', 2)
    project = _messages.StringField(3, required=True)
    zone = _messages.StringField(4, required=True)