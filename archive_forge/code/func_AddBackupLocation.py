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
def AddBackupLocation(parser, allow_empty, hidden=False):
    help_text = 'Choose where to store your backups. Backups are stored in the closest multi-region location to you by default. Only customize if needed.'
    if allow_empty:
        help_text += ' Specify empty string to revert to default.'
    parser.add_argument('--backup-location', required=False, help=help_text, hidden=hidden)