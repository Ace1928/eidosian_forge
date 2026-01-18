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
def AddScheduleTime(parser):
    """Add the flag for maintenance reschedule schedule time.

  Args:
    parser: The current argparse parser to add this to.
  """
    parser.add_argument('--schedule-time', type=arg_parsers.Datetime.Parse, help='When specifying SPECIFIC_TIME, the date and time at which to schedule the maintenance in ISO 8601 format.')