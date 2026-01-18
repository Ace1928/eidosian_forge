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
def AddBackupExpiryTime(parser):
    """Add the flag to specify the expiration time of the backup.

  Args:
    parser: The current parser to add this argument.
  """
    parser.add_argument('--expiry-time', required=False, type=arg_parsers.Datetime.Parse, default=None, hidden=True, help='Specifies when the final backup expires. The Maximum time allowed is 365 days from now. Format: YYYY-MM-DDTHH:MM:SS. Provide either ttl-days or expiry-time.')