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
def AddPasswordPolicyAllowedFailedAttempts(parser):
    """Add the flag to set number of failed login attempts allowed before a user is locked.

  Args:
    parser: The current argparse parser to add this to.
  """
    parser.add_argument('--password-policy-allowed-failed-attempts', type=int, required=False, default=None, help='Number of failed login attempts allowed before a user is locked out. This flag is available only for MySQL.')