from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RecaptchaenterpriseProjectsFirewallpoliciesDeleteRequest(_messages.Message):
    """A RecaptchaenterpriseProjectsFirewallpoliciesDeleteRequest object.

  Fields:
    name: Required. The name of the policy to be deleted, in the format
      `projects/{project}/firewallpolicies/{firewallpolicy}`.
  """
    name = _messages.StringField(1, required=True)