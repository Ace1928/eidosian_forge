from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddPartnerAsn(parser):
    """Adds partner asn flag to the argparse.ArgumentParser.

  Args:
    parser: The argparse parser.
  """
    parser.add_argument('--partner-asn', type=int, help='      BGP ASN of the partner. This should only be supplied by layer 3\n      providers that have configured BGP on behalf of the customer.\n      ')