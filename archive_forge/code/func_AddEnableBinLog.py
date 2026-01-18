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
def AddEnableBinLog(parser, show_negated_in_help=False, hidden=False):
    kwargs = _GetKwargsForBoolFlag(show_negated_in_help)
    parser.add_argument('--enable-bin-log', required=False, help='Allows for data recovery from a specific point in time, down to a fraction of a second. Must have automatic backups enabled to use. Make sure storage can support at least 7 days of logs.', hidden=hidden, **kwargs)