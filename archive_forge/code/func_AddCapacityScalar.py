from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.core import log
def AddCapacityScalar(parser, support_global_neg=False, support_region_neg=False):
    """Adds capacity thresholds argument to the argparse."""
    help_text = '      Scales down the target capacity (max utilization, max rate, or max\n      connections) without changing the target capacity. For usage guidelines\n      and examples, see\n      [Capacity scaler](https://cloud.google.com/load-balancing/docs/backend-service#capacity_scaler).\n      '
    incompatible_types = []
    if support_global_neg:
        incompatible_types.extend(['INTERNET_IP_PORT', 'INTERNET_FQDN_PORT'])
    if support_region_neg:
        incompatible_types.append('SERVERLESS')
    if incompatible_types:
        help_text += '\n    This cannot be used when the endpoint type of an attached network endpoint\n    group is {0}.\n    '.format(_JoinTypes(incompatible_types))
    parser.add_argument('--capacity-scaler', type=arg_parsers.BoundedFloat(lower_bound=0.0, upper_bound=1.0), help=help_text)