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
def AddAllowGlobalAccess(parser):
    """Adds allow global access flag to the argparse."""
    parser.add_argument('--allow-global-access', action='store_true', default=None, help='      If True, then clients from all regions can access this internal\n      forwarding rule. This can only be specified for forwarding rules with\n      the LOAD_BALANCING_SCHEME set to INTERNAL or INTERNAL_MANAGED. For\n      forwarding rules of type INTERNAL, the target must be either a backend\n      service or a target instance.\n      ')