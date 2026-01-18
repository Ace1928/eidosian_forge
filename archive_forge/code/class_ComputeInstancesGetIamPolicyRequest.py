from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeInstancesGetIamPolicyRequest(_messages.Message):
    """A ComputeInstancesGetIamPolicyRequest object.

  Fields:
    optionsRequestedPolicyVersion: Requested IAM Policy version.
    project: Project ID for this request.
    resource: Name or id of the resource for this request.
    zone: The name of the zone for this request.
  """
    optionsRequestedPolicyVersion = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    project = _messages.StringField(2, required=True)
    resource = _messages.StringField(3, required=True)
    zone = _messages.StringField(4, required=True)