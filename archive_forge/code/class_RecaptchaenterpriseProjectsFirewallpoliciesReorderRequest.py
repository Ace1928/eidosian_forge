from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RecaptchaenterpriseProjectsFirewallpoliciesReorderRequest(_messages.Message):
    """A RecaptchaenterpriseProjectsFirewallpoliciesReorderRequest object.

  Fields:
    googleCloudRecaptchaenterpriseV1ReorderFirewallPoliciesRequest: A
      GoogleCloudRecaptchaenterpriseV1ReorderFirewallPoliciesRequest resource
      to be passed as the request body.
    parent: Required. The name of the project to list the policies for, in the
      format `projects/{project}`.
  """
    googleCloudRecaptchaenterpriseV1ReorderFirewallPoliciesRequest = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1ReorderFirewallPoliciesRequest', 1)
    parent = _messages.StringField(2, required=True)