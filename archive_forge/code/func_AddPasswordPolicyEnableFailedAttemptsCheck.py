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
def AddPasswordPolicyEnableFailedAttemptsCheck(parser, show_negated_in_help=True):
    """Add the flag to enable the failed login attempts check.

  Args:
    parser: The current argparse parser to add this to.
    show_negated_in_help: Show nagative action in help.
  """
    kwargs = _GetKwargsForBoolFlag(show_negated_in_help)
    parser.add_argument('--password-policy-enable-failed-attempts-check', required=False, help='Enables the failed login attempts check if set to true. This flag is available only for MySQL.', **kwargs)