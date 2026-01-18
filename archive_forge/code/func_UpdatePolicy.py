from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from apitools.base.py import exceptions as api_exceptions
from googlecloudsdk.api_lib.orgpolicy import service as org_policy_service
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.org_policies import arguments
from googlecloudsdk.command_lib.org_policies import exceptions
from googlecloudsdk.command_lib.org_policies import utils
from googlecloudsdk.core import log
def UpdatePolicy(self, policy, policy_name, update_mask):
    """Update the policy on the service.

    Args:
      policy: messages.GoogleCloudOrgpolicy{api_version}Policy, The policy
        object to be updated.
      policy_name: Name of the policy to be updated
      update_mask: Specifies whether live/dryrun spec needs to be updated.

    Returns:
      Returns the updated policy.
    """
    updated_policy = self.ResetPolicy(policy, update_mask)
    if updated_policy == policy:
        return policy
    update_response = self.org_policy_api.UpdatePolicy(updated_policy, update_mask)
    log.UpdatedResource(policy_name, 'policy')
    return update_response