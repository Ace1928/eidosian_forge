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
def AddUserDiscardDualPassword(parser):
    """Will discard the user's secondary password.

  Args:
    parser: The current argparse parser to add this to.
  """
    kwargs = _GetKwargsForBoolFlag(False)
    parser.add_argument('--discard-dual-password', required=False, help="Discard the user's secondary password. Cannot set password and set this flag. This flag is only available for MySQL 8.0.", **kwargs)