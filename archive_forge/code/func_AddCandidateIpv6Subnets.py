from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddCandidateIpv6Subnets(parser):
    """Adds candidate ipv6 subnets flag to the argparse.ArgumentParser.

  Args:
    parser: The argparse parser.
  """
    parser.add_argument('--candidate-ipv6-subnets', type=arg_parsers.ArgList(max_length=16), metavar='IPV6_SUBNET', help='The `candididate-ipv6-subnets` field is not available.', default=[])