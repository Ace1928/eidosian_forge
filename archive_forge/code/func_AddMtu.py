from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddMtu(parser):
    """Adds mtu flag to the argparse.ArgumentParser.

  Args:
    parser: The argparse parser.
  """
    parser.add_argument('--mtu', type=int, help='      Maximum transmission unit (MTU) is the size of the largest IP packet\n      passing through this interconnect attachment. Only 1440 and 1500 are\n      allowed values. If not specified, the value will default to 1440.\n      ')