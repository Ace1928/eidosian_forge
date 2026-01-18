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
def MakeVmMaintenanceMaintenanceWindow(policy_ref, args, messages):
    """Creates a VM Maintenance window policy message from args."""
    vm_policy = messages.ResourcePolicyVmMaintenancePolicy()
    _, daily_cycle, _ = _ParseCycleFrequencyArgs(args, messages)
    vm_policy.maintenanceWindow = messages.ResourcePolicyVmMaintenancePolicyMaintenanceWindow(dailyMaintenanceWindow=daily_cycle)
    return messages.ResourcePolicy(name=policy_ref.Name(), description=args.description, region=policy_ref.region, vmMaintenancePolicy=vm_policy)