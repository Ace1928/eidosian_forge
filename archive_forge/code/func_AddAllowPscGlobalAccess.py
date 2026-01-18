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
def AddAllowPscGlobalAccess(parser):
    """Adds allow PSC global access flag to the argparse."""
    parser.add_argument('--allow-psc-global-access', action='store_true', default=None, help="      If specified, clients from all regions can access this Private\n      Service Connect forwarding rule. This can only be specified if the\n      forwarding rule's target is a service attachment\n      (--target-service-attachment).\n      ")