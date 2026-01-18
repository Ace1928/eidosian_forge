from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import scope as compute_scopes
from googlecloudsdk.command_lib.compute.resource_policies import flags
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import times
def MakeDiskConsistencyGroupPolicy(policy_ref, args, messages):
    """Creates a Disk Consistency Group Resource Policy message from args.

  Args:
    policy_ref: resource reference of the Disk Consistency Group policy.
    args: Namespace, argparse.Namespace.
    messages: message classes.

  Returns:
    A messages.ResourcePolicy object for Disk Consistency Group Resource Policy.
  """
    consistency_group_policy = messages.ResourcePolicyDiskConsistencyGroupPolicy()
    return messages.ResourcePolicy(name=policy_ref.Name(), description=args.description, region=policy_ref.region, diskConsistencyGroupPolicy=consistency_group_policy)