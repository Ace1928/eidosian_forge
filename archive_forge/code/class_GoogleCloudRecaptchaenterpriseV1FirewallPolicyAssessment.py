from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1FirewallPolicyAssessment(_messages.Message):
    """Policy config assessment.

  Fields:
    error: Output only. If the processing of a policy config fails, an error
      will be populated and the firewall_policy will be left empty.
    firewallPolicy: Output only. The policy that matched the request. If more
      than one policy may match, this is the first match. If no policy matches
      the incoming request, the policy field will be left empty.
  """
    error = _messages.MessageField('GoogleRpcStatus', 1)
    firewallPolicy = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1FirewallPolicy', 2)