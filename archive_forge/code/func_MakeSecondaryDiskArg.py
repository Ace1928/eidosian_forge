from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.core import properties
def MakeSecondaryDiskArg(required=False):
    return compute_flags.ResourceArgument(resource_name='async secondary disk', name='--secondary-disk', completer=compute_completers.DisksCompleter, zonal_collection='compute.disks', regional_collection='compute.regionDisks', short_help='Secondary disk for asynchronous replication.', detailed_help=_ASYNC_SECONDARY_DISK_HELP, plural=False, required=required, scope_flags_usage=compute_flags.ScopeFlagsUsage.GENERATE_DEDICATED_SCOPE_FLAGS, zone_help_text=_ASYNC_SECONDARY_DISK_ZONE_EXPLANATION, region_help_text=_ASYNC_SECONDARY_DISK_REGION_EXPLANATION)