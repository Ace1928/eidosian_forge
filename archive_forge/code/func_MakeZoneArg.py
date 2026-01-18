from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def MakeZoneArg():
    return compute_flags.ResourceArgument(resource_name='zone', completer=compute_completers.ZonesCompleter, plural=False, required=True, global_collection='compute.zones')