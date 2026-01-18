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
def CreatePolicy(self, policy_name, update_mask):
    """Create the policy on the service if needed.

    Args:
      policy_name: Name of the policy to be created
      update_mask: Specifies whether live/dryrun spec needs to be created.

    Returns:
      The created policy.
    """
    empty_policy = self.org_policy_api.BuildEmptyPolicy(policy_name)
    new_policy = self.ResetPolicy(empty_policy, update_mask)
    create_response = self.org_policy_api.CreatePolicy(new_policy)
    log.CreatedResource(policy_name, 'policy')
    return create_response