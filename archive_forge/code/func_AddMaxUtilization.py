from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.core import log
def AddMaxUtilization(parser):
    """Adds max utilization argument to the argparse."""
    parser.add_argument('--max-utilization', type=arg_parsers.BoundedFloat(lower_bound=0.0, upper_bound=1.0), help='      Defines the maximum target for average utilization of the backend instance\n      group. Supported values are `0.0` (0%) through `1.0` (100%). This is an\n      optional parameter for the `UTILIZATION` balancing mode.\n\n      You can use this parameter with other parameters for defining target\n      capacity. For usage guidelines, see\n      [Balancing mode combinations](https://cloud.google.com/load-balancing/docs/backend-service#balancing-mode-combos).\n      ')