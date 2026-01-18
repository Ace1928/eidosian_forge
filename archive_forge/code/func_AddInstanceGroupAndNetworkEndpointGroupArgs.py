from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.network_services import completers as network_services_completers
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddInstanceGroupAndNetworkEndpointGroupArgs(parser, verb, support_global_neg=False, support_region_neg=False):
    """Adds instance group and network endpoint group args to the argparse."""
    backend_group = parser.add_group(required=True, mutex=True)
    instance_group = backend_group.add_group('Instance Group')
    neg_group = backend_group.add_group('Network Endpoint Group')
    MULTISCOPE_INSTANCE_GROUP_ARG.AddArgument(instance_group, operation_type='{} the backend service'.format(verb))
    neg_group_arg = GetNetworkEndpointGroupArg(support_global_neg=support_global_neg, support_region_neg=support_region_neg)
    neg_group_arg.AddArgument(neg_group, operation_type='{} the backend service'.format(verb))