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
def AddFinalBackupExpiryTimeArgument(parser):
    parser.add_argument('--final-backup-expiry-time', type=arg_parsers.Datetime.Parse, required=False, hidden=True, help='Specifies the time at which the final backup will expire. Maximum time allowed is 365 days from now. Format: YYYY-MM-DDTHH:MM:SS.')