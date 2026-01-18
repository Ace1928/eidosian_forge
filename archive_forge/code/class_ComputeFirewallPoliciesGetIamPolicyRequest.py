from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeFirewallPoliciesGetIamPolicyRequest(_messages.Message):
    """A ComputeFirewallPoliciesGetIamPolicyRequest object.

  Fields:
    optionsRequestedPolicyVersion: Requested IAM Policy version.
    resource: Name or id of the resource for this request.
  """
    optionsRequestedPolicyVersion = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    resource = _messages.StringField(2, required=True)