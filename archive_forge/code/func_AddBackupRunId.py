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
def AddBackupRunId(parser):
    """Add the flag for ID of backup run.

  Args:
    parser: The current argparse parser to add this to.
  """
    parser.add_argument('id', type=arg_parsers.BoundedInt(lower_bound=1, unlimited=True), help='The ID of the backup run. You can find the ID by running $ gcloud sql backups list -i {instance}.')