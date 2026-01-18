from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddSimulateMaintenanceEventNodesArgToParser(parser):
    """Add --nodes flag."""
    parser.add_argument('--nodes', metavar='NODE', type=arg_parsers.ArgList(), help='The names of the nodes to simulate maintenance event.')