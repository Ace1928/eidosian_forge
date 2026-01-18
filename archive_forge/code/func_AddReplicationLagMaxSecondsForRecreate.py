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
def AddReplicationLagMaxSecondsForRecreate(parser):
    """Adds the '--replication-lag-max-seconds-for-recreate' flag to the parser for instances patch action.

  Args:
    parser: The current argparse parser to add this to.
  """
    parser.add_argument('--replication-lag-max-seconds-for-recreate', type=arg_parsers.BoundedInt(lower_bound=300, upper_bound=31536000), hidden=True, action=arg_parsers.StoreOnceAction, required=False, help='Set a maximum replication lag for a read replica inseconds, If the replica lag exceeds the specified value, the readreplica(s) will be recreated. Min value=300 seconds,Max value=31536000 seconds.')