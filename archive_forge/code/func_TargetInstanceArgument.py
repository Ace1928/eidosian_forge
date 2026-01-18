from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def TargetInstanceArgument(required=True, plural=False):
    return compute_flags.ResourceArgument(resource_name='target instance', completer=TargetInstancesCompleter, plural=plural, required=required, zonal_collection='compute.targetInstances', zone_explanation=compute_flags.ZONE_PROPERTY_EXPLANATION)