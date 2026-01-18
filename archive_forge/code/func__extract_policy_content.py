from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.container.fleet.policycontroller import protos
from googlecloudsdk.command_lib.container.fleet.policycontroller import exceptions
from googlecloudsdk.command_lib.container.fleet.policycontroller import flags
def _extract_policy_content(self, poco_cfg) -> messages.Message:
    """Gets the PolicyControllerPolicyContentSpec message from the hub config.

    Args:
      poco_cfg: The MembershipFeatureSpec message.

    Returns:
      The PolicyControllerPolicyContentSpec message or an empty one if not
      found.
    """
    if poco_cfg.policycontroller.policyControllerHubConfig.policyContent is None:
        return self.messages.PolicyControllerPolicyContentSpec()
    return poco_cfg.policycontroller.policyControllerHubConfig.policyContent