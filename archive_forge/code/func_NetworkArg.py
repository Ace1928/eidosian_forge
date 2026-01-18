from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.addresses import flags as addresses_flags
from googlecloudsdk.command_lib.util import completers
def NetworkArg():
    """Returns the network parameter."""
    load_balancing_scheme = '--load-balancing-scheme=INTERNAL or --load-balancing-scheme=INTERNAL_SELF_MANAGED or --load-balancing-scheme=EXTERNAL_MANAGED (regional) or --load-balancing-scheme=INTERNAL_MANAGED'
    return compute_flags.ResourceArgument(name='--network', required=False, resource_name='network', global_collection='compute.networks', short_help='Network that this forwarding rule applies to.', detailed_help='\n          (Only for %s) Network that this\n          forwarding rule applies to. If this field is not specified, the default\n          network is used. In the absence of the default network, this field\n          must be specified.\n          ' % load_balancing_scheme)