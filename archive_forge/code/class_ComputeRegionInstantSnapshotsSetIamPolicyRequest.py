from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeRegionInstantSnapshotsSetIamPolicyRequest(_messages.Message):
    """A ComputeRegionInstantSnapshotsSetIamPolicyRequest object.

  Fields:
    project: Project ID for this request.
    region: The name of the region for this request.
    regionSetPolicyRequest: A RegionSetPolicyRequest resource to be passed as
      the request body.
    resource: Name or id of the resource for this request.
  """
    project = _messages.StringField(1, required=True)
    region = _messages.StringField(2, required=True)
    regionSetPolicyRequest = _messages.MessageField('RegionSetPolicyRequest', 3)
    resource = _messages.StringField(4, required=True)