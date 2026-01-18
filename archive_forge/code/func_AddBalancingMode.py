from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.core import log
def AddBalancingMode(parser, support_global_neg=False, support_region_neg=False):
    """Adds balancing mode argument to the argparse."""
    help_text = '  Defines how to measure whether a backend can handle additional traffic or is\n  fully loaded. For more information, see\n  https://cloud.google.com/load-balancing/docs/backend-service#balancing-mode.\n  '
    incompatible_types = []
    if support_global_neg:
        incompatible_types.extend(['INTERNET_IP_PORT', 'INTERNET_FQDN_PORT'])
    if support_region_neg:
        incompatible_types.append('SERVERLESS')
    if incompatible_types:
        help_text += '\n  This cannot be used when the endpoint type of an attached network endpoint\n  group is {0}.\n    '.format(_JoinTypes(incompatible_types))
    parser.add_argument('--balancing-mode', choices=_GetBalancingModes(), type=lambda x: x.upper(), help=help_text)