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
def AddDeletionProtection(parser, hidden=False):
    """Adds the '--deletion-protection' flag to the parser for instances patch action.

  Args:
    parser: The current argparse parser to add this to.
    hidden: if the field needs to be hidden.
  """
    help_text = 'Enable deletion protection on a Cloud SQL instance.'
    parser.add_argument('--deletion-protection', action=arg_parsers.StoreTrueFalseAction, help=help_text, hidden=hidden)