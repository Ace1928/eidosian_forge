from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def MakeMachineTypeArg():
    return compute_flags.ResourceArgument(resource_name='machine type', completer=completers.MachineTypesCompleter, zonal_collection='compute.machineTypes', zone_explanation=compute_flags.ZONE_PROPERTY_EXPLANATION)