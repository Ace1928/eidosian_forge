from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
def AddBackupFlag(parser):
    """Adds flag for backup to the given parser.

  Args:
    parser: The argparse parser.
  """
    parser.add_argument('--backup', metavar='BACKUP', required=True, type=str, help="\n      The backup to operate on.\n\n      For example, to operate on backup `cf9f748a-7980-4703-b1a1-d1ffff591db0`:\n\n        $ {command} --backup='cf9f748a-7980-4703-b1a1-d1ffff591db0'\n      ")