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
def AddDatabaseList(parser, help_text, required=False):
    """Add the '--database' and '-d' list flags to the parser.

  Args:
    parser: The current argparse parser to add these database flags to.
    help_text: String, specifies the help text for the database flags.
    required: Boolean, specifies whether the database flag is required.
  """
    if required:
        group = parser.add_group(mutex=False, required=True)
        group.add_argument('--database', '-d', type=arg_parsers.ArgList(min_length=1), metavar='DATABASE', help=help_text)
    else:
        parser.add_argument('--database', '-d', type=arg_parsers.ArgList(min_length=1), metavar='DATABASE', required=False, help=help_text)