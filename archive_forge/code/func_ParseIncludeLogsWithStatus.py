from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.builds import flags as build_flags
def ParseIncludeLogsWithStatus(trigger, args, messages):
    """Parses include logs with status flag.

  Args:
    trigger: The trigger to populate.
    args: An argparse arguments object.
    messages: A Cloud Build messages module.
  """
    if args.include_logs_with_status:
        trigger.includeBuildLogs = messages.BuildTrigger.IncludeBuildLogsValueValuesEnum.INCLUDE_BUILD_LOGS_WITH_STATUS