from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.compute import utils as compute_utils
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util import completers
def AddPasswordPolicyMinLength(parser, hidden=False):
    """Add the flag to specify password policy min length.

  Args:
    parser: The current argparse parser to add this to.
    hidden: if the field needs to be hidden.
  """
    parser.add_argument('--password-policy-min-length', type=int, required=False, default=None, help='Minimum number of characters allowed in the password.', hidden=hidden)