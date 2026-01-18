from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeFirewallPoliciesSetIamPolicyRequest(_messages.Message):
    """A ComputeFirewallPoliciesSetIamPolicyRequest object.

  Fields:
    globalOrganizationSetPolicyRequest: A GlobalOrganizationSetPolicyRequest
      resource to be passed as the request body.
    resource: Name or id of the resource for this request.
  """
    globalOrganizationSetPolicyRequest = _messages.MessageField('GlobalOrganizationSetPolicyRequest', 1)
    resource = _messages.StringField(2, required=True)