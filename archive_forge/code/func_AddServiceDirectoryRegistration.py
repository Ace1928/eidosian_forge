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
def AddServiceDirectoryRegistration(parser):
    """Adds service-directory-registration flag to the argparse."""
    parser.add_argument('--service-directory-registration', type=str, action='store', default=None, help='      The Service Directory service in which to register this forwarding rule as\n      an endpoint. The Service Directory service must be in the same project and\n      region as the forwarding rule you are creating.\n      ')