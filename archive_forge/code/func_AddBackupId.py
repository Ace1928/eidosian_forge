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
def AddBackupId(parser, help_text='The ID of the backup run. To find the ID, run the following command: $ gcloud sql backups list -i {instance}.'):
    """Add the flag for the ID of the backup run.

  Args:
    parser: The current argparse parser to which to add this.
    help_text: The help text to display.
  """
    parser.add_argument('id', help=help_text)