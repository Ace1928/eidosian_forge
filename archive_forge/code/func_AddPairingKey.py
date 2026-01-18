from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddPairingKey(parser):
    """Adds pairing key flag to the argparse.ArgumentParser.

  Args:
    parser: The argparse parser.
  """
    parser.add_argument('--pairing-key', required=True, help='      Value of the pairing-key from the target partner attachment provided by\n      the customer.\n      ')