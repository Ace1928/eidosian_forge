from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeInstanceGroupManagersGetRequest(_messages.Message):
    """A ComputeInstanceGroupManagersGetRequest object.

  Fields:
    instanceGroupManager: The name of the managed instance group.
    project: Project ID for this request.
    zone: The name of the zone where the managed instance group is located.
  """
    instanceGroupManager = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)
    zone = _messages.StringField(3, required=True)