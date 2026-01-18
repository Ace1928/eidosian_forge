from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def MakeInstantSnapshotArg(plural=False):
    return compute_flags.ResourceArgument(resource_name='instant snapshot', completer=compute_completers.InstantSnapshotsCompleter, plural=plural, name='INSTANT_SNAPSHOT_NAME', zonal_collection='compute.instantSnapshots', regional_collection='compute.regionInstantSnapshots', zone_explanation=compute_flags.ZONE_PROPERTY_EXPLANATION, region_explanation=compute_flags.REGION_PROPERTY_EXPLANATION)