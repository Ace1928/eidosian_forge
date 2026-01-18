from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeInstantSnapshotsSetIamPolicyRequest(_messages.Message):
    """A ComputeInstantSnapshotsSetIamPolicyRequest object.

  Fields:
    project: Project ID for this request.
    resource: Name or id of the resource for this request.
    zone: The name of the zone for this request.
    zoneSetPolicyRequest: A ZoneSetPolicyRequest resource to be passed as the
      request body.
  """
    project = _messages.StringField(1, required=True)
    resource = _messages.StringField(2, required=True)
    zone = _messages.StringField(3, required=True)
    zoneSetPolicyRequest = _messages.MessageField('ZoneSetPolicyRequest', 4)