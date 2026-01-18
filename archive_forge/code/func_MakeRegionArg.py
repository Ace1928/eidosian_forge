from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def MakeRegionArg():
    return compute_flags.ResourceArgument(resource_name='region', completer=compute_completers.RegionsCompleter, global_collection='compute.regions')