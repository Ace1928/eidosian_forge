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
def AddClearFailoverDrReplicaName(parser, hidden=False):
    kwargs = _GetKwargsForBoolFlag(False)
    parser.add_argument('--clear-failover-dr-replica-name', required=False, hidden=hidden, help='Clear the DR replica setting for the primary instance. Flag is only available for MySQL.', **kwargs)