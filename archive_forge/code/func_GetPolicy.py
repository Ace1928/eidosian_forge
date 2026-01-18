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
def GetPolicy(self, policy_name):
    """Get the policy from the service.

    Args:
      policy_name: Name of the policy to be retrieved.

    Returns:
      The retrieved policy, or None if not found.
    """
    try:
        return self.org_policy_api.GetPolicy(policy_name)
    except api_exceptions.HttpNotFoundError:
        return None