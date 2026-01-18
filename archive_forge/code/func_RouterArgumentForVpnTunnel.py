from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def RouterArgumentForVpnTunnel(required=True):
    return compute_flags.ResourceArgument(resource_name='router', name='--router', completer=RoutersCompleter, plural=False, required=required, regional_collection='compute.routers', short_help='The Router to use for dynamic routing.', region_explanation=compute_flags.REGION_PROPERTY_EXPLANATION)