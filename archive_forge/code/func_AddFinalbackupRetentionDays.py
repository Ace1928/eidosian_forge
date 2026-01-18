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
def AddFinalbackupRetentionDays(parser):
    help_text = 'Specifies number of days to retain final backup. The valid range is between 1 and 365. Default value is 30 days.'
    parser.add_argument('--final-backup-retention-days', type=arg_parsers.BoundedInt(1, 365, unlimited=False), required=False, help=help_text, hidden=True, default=30)