from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.auth import external_account as auth_external_account
from googlecloudsdk.api_lib.auth import service_account as auth_service_account
from googlecloudsdk.api_lib.auth import util as auth_util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.auth import auth_util as command_auth_util
from googlecloudsdk.command_lib.auth import flags as auth_flags
from googlecloudsdk.command_lib.auth import workforce_login_config as workforce_login_config_util
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.credentials import devshell as c_devshell
from googlecloudsdk.core.credentials import exceptions as creds_exceptions
from googlecloudsdk.core.credentials import gce as c_gce
from googlecloudsdk.core.credentials import store as c_store
def ShouldContinueLogin(cred_config):
    """Prompts users if they try to login in managed environments.

  Args:
    cred_config: Json object loaded from the file specified in --cred-file.

  Returns:
    True if users want to continue the login command.
  """
    if c_devshell.IsDevshellEnvironment():
        message = textwrap.dedent('\n          You are already authenticated with gcloud when running\n          inside the Cloud Shell and so do not need to run this\n          command. Do you wish to proceed anyway?\n          ')
        return console_io.PromptContinue(message=message)
    elif c_gce.Metadata().connected and (not auth_external_account.IsExternalAccountConfig(cred_config)):
        message = textwrap.dedent('\n          You are running on a Google Compute Engine virtual machine.\n          It is recommended that you use service accounts for authentication.\n\n          You can run:\n\n            $ gcloud config set account `ACCOUNT`\n\n          to switch accounts if necessary.\n\n          Your credentials may be visible to others with access to this\n          virtual machine. Are you sure you want to authenticate with\n          your personal account?\n          ')
        return console_io.PromptContinue(message=message)
    else:
        return True