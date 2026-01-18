from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.builds import flags as build_flags
def AddIncludeLogsArgs(flag_config):
    """Add flag related to including logs for GitHub checkrun summary page.

  Args:
    flag_config: argparse argument group. Include logs for GitHub will be
    added to this config.
  """
    flag_config.add_argument('--include-logs-with-status', help='Build logs will be sent back to GitHub as part of the checkrun result.', action='store_true')