from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
def AddDatabaseIdFlag(parser, required=False):
    """Adds flag for database id to the given parser.

  Args:
    parser: The argparse parser.
    required: Whether the flag must be set for running the command, a bool.
  """
    if not required:
        helper_text = "      The database to operate on. The default value is `(default)`.\n\n      For example, to operate on database `foo`:\n\n        $ {command} --database='foo'\n      "
    else:
        helper_text = "      The database to operate on.\n\n      For example, to operate on database `foo`:\n\n        $ {command} --database='foo'\n      "
    parser.add_argument('--database', metavar='DATABASE', type=str, default='(default)' if not required else None, required=required, help=helper_text)