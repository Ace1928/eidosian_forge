from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
def AddRetentionFlag(parser, required=False):
    """Adds flag for retention to the given parser.

  Args:
    parser: The argparse parser.
    required: Whether the flag must be set for running the command, a bool.
  """
    parser.add_argument('--retention', metavar='RETENTION', required=required, type=arg_parsers.Duration(), help=textwrap.dedent('          The rention of the backup. At what relative time in the future,\n          compared to the creation time of the backup should the backup be\n          deleted, i.e. keep backups for 7 days.\n\n          For example, to set retention as 7 days.\n\n          $ {command} --retention=7d\n          '))