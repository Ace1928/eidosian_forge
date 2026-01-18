from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def MakeSourceInstanceArg():
    return compute_flags.ResourceArgument(resource_name='instance', name='--source-instance', completer=compute_completers.InstancesCompleter, required=True, zonal_collection='compute.instances', short_help='The source instance to create a machine image from.', zone_explanation=compute_flags.ZONE_PROPERTY_EXPLANATION)