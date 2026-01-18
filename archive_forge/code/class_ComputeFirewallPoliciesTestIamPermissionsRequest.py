from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeFirewallPoliciesTestIamPermissionsRequest(_messages.Message):
    """A ComputeFirewallPoliciesTestIamPermissionsRequest object.

  Fields:
    resource: Name or id of the resource for this request.
    testPermissionsRequest: A TestPermissionsRequest resource to be passed as
      the request body.
  """
    resource = _messages.StringField(1, required=True)
    testPermissionsRequest = _messages.MessageField('TestPermissionsRequest', 2)