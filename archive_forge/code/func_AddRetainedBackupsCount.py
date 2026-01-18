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
def AddRetainedBackupsCount(parser, hidden=False):
    help_text = 'How many backups to keep. The valid range is between 1 and 365. Default value is 7 for Enterprise edition instances. For Enterprise_Plus, default value is 15. Applicable only if --no-backups is not specified.'
    parser.add_argument('--retained-backups-count', type=arg_parsers.BoundedInt(1, 365, unlimited=False), help=help_text, hidden=hidden)