from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
def AddBackupScheduleFlag(parser):
    """Adds flag for backup schedule id to the given parser.

  Args:
    parser: The argparse parser.
  """
    parser.add_argument('--backup-schedule', metavar='BACKUP_SCHEDULE', required=True, type=str, help="\n      The backup schedule to operate on.\n\n      For example, to operate on backup schedule `091a49a0-223f-4c98-8c69-a284abbdb26b`:\n\n        $ {command} --backup-schedule='091a49a0-223f-4c98-8c69-a284abbdb26b'\n      ")