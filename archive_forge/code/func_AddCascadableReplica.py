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
def AddCascadableReplica(parser, hidden=False):
    """Adds --cascadable-replica flag."""
    kwargs = _GetKwargsForBoolFlag(False)
    parser.add_argument('--cascadable-replica', required=False, help='Specifies whether a SQL Server replica is a cascadable replica. A cascadable replica is a SQL Server cross-region replica that supports replica(s) under it. This flag only takes effect when the `--master-instance-name` flag is set, and the replica under creation is in a different region than the primary instance.', hidden=hidden, **kwargs)