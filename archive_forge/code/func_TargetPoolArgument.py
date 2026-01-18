from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def TargetPoolArgument(required=True, help_suffix='.', plural=False):
    return compute_flags.ResourceArgument(resource_name='target pool', completer=TargetPoolsCompleter, plural=plural, required=required, regional_collection='compute.targetPools', short_help=help_suffix and 'The name of the target pool{0}'.format(help_suffix), region_explanation=compute_flags.REGION_PROPERTY_EXPLANATION)