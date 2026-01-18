from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddCreateSubnetModeArg(parser):
    """Adds the --subnet-mode flag."""
    parser.add_argument('--subnet-mode', choices=_CREATE_SUBNET_MODE_CHOICES, type=lambda mode: mode.lower(), metavar='MODE', help='The subnet mode of the network. If not specified, defaults to\n              AUTO.')