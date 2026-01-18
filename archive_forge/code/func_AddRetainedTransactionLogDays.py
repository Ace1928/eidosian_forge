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
def AddRetainedTransactionLogDays(parser, hidden=False):
    help_text = 'How many days of transaction logs to keep. The valid range is between 1 and 35. Only use this option when point-in-time recovery is enabled. If logs are stored on disk, storage size for transaction logs could increase when the number of days for log retention increases. For Enterprise, default and max retention values are 7 and 7 respectively. For Enterprise_Plus, default and max retention values are 14 and 35.'
    parser.add_argument('--retained-transaction-log-days', type=arg_parsers.BoundedInt(1, 35, unlimited=False), help=help_text, hidden=hidden)