from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddCreateBgpRoutingModeArg(parser):
    """Adds the --bgp-routing-mode flag."""
    parser.add_argument('--bgp-routing-mode', choices=_BGP_ROUTING_MODE_CHOICES, default='regional', type=lambda mode: mode.lower(), metavar='MODE', help='The BGP routing mode for this network. If not specified, defaults\n              to regional.')