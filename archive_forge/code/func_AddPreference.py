from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.core import log
def AddPreference(parser):
    """Adds preference argument to the argparse."""
    help_text = '  Defines whether a backend should be fully utilized before\n  sending traffic to backends with default preference.\n  '
    incompatible_types = ['INTERNET_IP_PORT', 'INTERNET_FQDN_PORT', 'SERVERLESS']
    help_text += '  This parameter cannot be used with regional managed instance groups and when\n  the endpoint type of an attached network endpoint group is {0}.\n  '.format(_JoinTypes(incompatible_types))
    parser.add_argument('--preference', choices=_GetPreference(), type=lambda x: x.upper(), help=help_text)