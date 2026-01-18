from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddStackType(parser):
    """Adds stack-type flag to the argparse.ArgumentParser.

  Args:
    parser: The argparse parser.
  """
    parser.add_argument('--stack-type', choices={'IPV4_ONLY': 'Only IPv4 protocol is enabled on this attachment.', 'IPV4_IPV6': 'Both IPv4 and IPv6 protocols are enabled on this attachment.'}, type=arg_utils.ChoiceToEnumName, help='The stack type of the protocol(s) enabled on this interconnect attachment.')