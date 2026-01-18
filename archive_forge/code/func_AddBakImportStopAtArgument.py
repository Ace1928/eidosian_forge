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
def AddBakImportStopAtArgument(parser):
    """Add the 'stop-at' argument to the parser for bak import."""
    parser.add_argument('--stop-at', type=arg_parsers.Datetime.Parse, required=False, help='Equivalent to SQL Server STOPAT keyword. Used in transaction log import only. Transaction log import stop at this timestamp. Format: YYYY-MM-DDTHH:MM:SS.')