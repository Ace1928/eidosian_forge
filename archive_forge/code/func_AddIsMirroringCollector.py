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
def AddIsMirroringCollector(parser):
    parser.add_argument('--is-mirroring-collector', action='store_true', default=None, help='      If set, this forwarding rule can be used as a collector for packet\n      mirroring. This can only be specified for forwarding rules with the\n      LOAD_BALANCING_SCHEME set to INTERNAL.\n      ')