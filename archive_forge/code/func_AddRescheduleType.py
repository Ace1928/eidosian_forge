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
def AddRescheduleType(parser):
    """Add the flag to specify reschedule type.

  Args:
    parser: The current argparse parser to add this to.
  """
    choices = [messages.Reschedule.RescheduleTypeValueValuesEnum.IMMEDIATE.name, messages.Reschedule.RescheduleTypeValueValuesEnum.NEXT_AVAILABLE_WINDOW.name, messages.Reschedule.RescheduleTypeValueValuesEnum.SPECIFIC_TIME.name]
    help_text = 'The type of reschedule operation to perform.'
    parser.add_argument('--reschedule-type', choices=choices, required=True, help=help_text)