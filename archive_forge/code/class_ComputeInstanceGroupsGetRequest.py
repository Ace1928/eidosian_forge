from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeInstanceGroupsGetRequest(_messages.Message):
    """A ComputeInstanceGroupsGetRequest object.

  Fields:
    instanceGroup: The name of the instance group.
    project: Project ID for this request.
    zone: The name of the zone where the instance group is located.
  """
    instanceGroup = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)
    zone = _messages.StringField(3, required=True)