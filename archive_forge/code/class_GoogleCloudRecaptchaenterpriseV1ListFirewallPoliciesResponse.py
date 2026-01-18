from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1ListFirewallPoliciesResponse(_messages.Message):
    """Response to request to list firewall policies belonging to a project.

  Fields:
    firewallPolicies: Policy details.
    nextPageToken: Token to retrieve the next page of results. It is set to
      empty if no policies remain in results.
  """
    firewallPolicies = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1FirewallPolicy', 1, repeated=True)
    nextPageToken = _messages.StringField(2)