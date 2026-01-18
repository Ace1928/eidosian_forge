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
def AddPortsAndPortRange(parser):
    """Adds ports and port range flags."""
    ports_scope = parser.add_mutually_exclusive_group()
    ports_metavar = 'ALL | [PORT | START_PORT-END_PORT],[...]'
    ports_help = '  List of comma-separated ports. The forwarding rule forwards packets with\n  matching destination ports. Port specification requirements vary\n  depending on the load-balancing scheme and target.\n  For more information, refer to https://cloud.google.com/load-balancing/docs/forwarding-rule-concepts#port_specifications.\n  '
    ports_scope.add_argument('--ports', metavar=ports_metavar, type=PortRangesWithAll.CreateParser(), default=None, help=ports_help)
    ports_scope.add_argument('--port-range', type=arg_parsers.Range.Parse, metavar='[PORT | START_PORT-END_PORT]', help='      DEPRECATED, use --ports. If specified, only packets addressed to ports in\n      the specified range are forwarded. For more information, refer to\n      https://cloud.google.com/load-balancing/docs/forwarding-rule-concepts#port_specifications.\n      ')