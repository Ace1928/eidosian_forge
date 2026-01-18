from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeInstanceGroupManagersApplyUpdatesToInstancesRequest(_messages.Message):
    """A ComputeInstanceGroupManagersApplyUpdatesToInstancesRequest object.

  Fields:
    instanceGroupManager: The name of the managed instance group, should
      conform to RFC1035.
    instanceGroupManagersApplyUpdatesRequest: A
      InstanceGroupManagersApplyUpdatesRequest resource to be passed as the
      request body.
    project: Project ID for this request.
    zone: The name of the zone where the managed instance group is located.
      Should conform to RFC1035.
  """
    instanceGroupManager = _messages.StringField(1, required=True)
    instanceGroupManagersApplyUpdatesRequest = _messages.MessageField('InstanceGroupManagersApplyUpdatesRequest', 2)
    project = _messages.StringField(3, required=True)
    zone = _messages.StringField(4, required=True)