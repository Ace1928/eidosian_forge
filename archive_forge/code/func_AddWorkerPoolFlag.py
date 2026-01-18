from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
import six
def AddWorkerPoolFlag(parser, hidden=False):
    """Adds a flag to send the build to a workerpool.

  Args:
    parser: The argparse parser to add the arg to.
    hidden: If true, retain help but do not display it.

  Returns:
    worker pool flag group
  """
    worker_pools = parser.add_argument_group('Worker pool only flags.')
    worker_pools.add_argument('--worker-pool', hidden=hidden, help='Specify a worker pool for the build to run in. Format: projects/{project}/locations/{region}/workerPools/{workerPool}.')
    return worker_pools