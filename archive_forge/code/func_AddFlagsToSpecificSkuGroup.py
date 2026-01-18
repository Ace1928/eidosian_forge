from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.compute.reservations import flags as reservation_flags
from googlecloudsdk.command_lib.compute.reservations import resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddFlagsToSpecificSkuGroup(group, support_stable_fleet=False):
    """Adds flags needed for a specific sku zonal allocation."""
    args = [reservation_flags.GetRequireSpecificAllocation(), reservation_flags.GetVmCountFlag(required=False), reservation_flags.GetMinCpuPlatform(), reservation_flags.GetMachineType(required=False), reservation_flags.GetLocalSsdFlag(), reservation_flags.GetAcceleratorFlag(), reservation_flags.GetResourcePolicyFlag()]
    if support_stable_fleet:
        args.append(instance_flags.AddMaintenanceInterval())
    for arg in args:
        arg.AddToParser(group)