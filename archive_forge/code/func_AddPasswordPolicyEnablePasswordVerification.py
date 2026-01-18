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
def AddPasswordPolicyEnablePasswordVerification(parser, show_negated_in_help=True):
    """Add the flag to specify password policy password verification.

  Args:
    parser: The current argparse parser to add this to.
    show_negated_in_help: Show nagative action in help.
  """
    kwargs = _GetKwargsForBoolFlag(show_negated_in_help)
    parser.add_argument('--password-policy-enable-password-verification', required=False, help='The current password must be specified when altering the password. This flag is available only for MySQL.', **kwargs)