from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.builds import flags as build_flags
def AddFlagConfigArgs(flag_config, add_name=True):
    """Adds additional argparse flags related to flag config.

  Args:
    flag_config: Argparse argument group. Additional flags will be added to this
      group to cover common flag configuration settings.
    add_name: If true, the trigger name is added.
  """
    if add_name:
        flag_config.add_argument('--name', help='Build trigger name.')
    flag_config.add_argument('--description', help='Build trigger description.')
    flag_config.add_argument('--service-account', help='The service account used for all user-controlled operations including UpdateBuildTrigger, RunBuildTrigger, CreateBuild, and CancelBuild. If no service account is set, then the standard Cloud Build service account ([PROJECT_NUM]@system.gserviceaccount.com) is used instead. Format: `projects/{PROJECT_ID}/serviceAccounts/{ACCOUNT_ID_OR_EMAIL}`.', required=False)
    flag_config.add_argument('--require-approval', help='Require manual approval for triggered builds.', action=arg_parsers.StoreTrueFalseAction)