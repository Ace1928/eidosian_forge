from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.builds import flags as build_flags
def AddGitLabEnterpriseTriggerArgs(parser):
    """Set up the generic argparse flags for creating or updating a build trigger for GitLab Enterprise.

  Args:
    parser: An argparse.ArgumentParser-like object.

  Returns:
    An empty parser group to be populated with flags specific to a trigger-type.
  """
    parser.display_info.AddFormat("\n          table(\n            name,\n            createTime.date('%Y-%m-%dT%H:%M:%S%Oz', undefined='-'),\n            status\n          )\n        ")
    trigger_config = parser.add_mutually_exclusive_group(required=True)
    AddTriggerConfigFilePathArg(trigger_config)
    flag_config = trigger_config.add_argument_group(help='Flag based trigger configuration')
    build_flags.AddRegionFlag(flag_config, hidden=False, required=False)
    AddFlagConfigArgs(flag_config)
    return flag_config