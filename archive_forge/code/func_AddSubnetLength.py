from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddSubnetLength(parser):
    """Adds subnet length flag to the argparse.ArgumentParser.

  Args:
    parser: The argparse parser.
  """
    parser.add_argument('--subnet-length', metavar='SUBNET_LENGTH', type=int, choices=frozenset({29, 30}), help='      The length of the IPv4 subnet mask for this attachment. 29 is the\n      default value, except for attachments on Cross-Cloud Interconnects whose\n      remote location\'s "constraints.subnetLengthRange" field specifies a\n      minimum subnet length of 30. In that case, the default value is 30.\n      The default value is recommended when there\'s no requirement on the subnet\n      length.\n      ')