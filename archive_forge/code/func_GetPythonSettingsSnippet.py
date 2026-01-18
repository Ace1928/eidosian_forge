from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import json
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.api_lib.auth import service_account
from googlecloudsdk.command_lib.artifacts import requests as ar_requests
from googlecloudsdk.command_lib.artifacts import util as ar_util
from googlecloudsdk.command_lib.artifacts.print_settings import apt
from googlecloudsdk.command_lib.artifacts.print_settings import gradle
from googlecloudsdk.command_lib.artifacts.print_settings import mvn
from googlecloudsdk.command_lib.artifacts.print_settings import npm
from googlecloudsdk.command_lib.artifacts.print_settings import python
from googlecloudsdk.command_lib.artifacts.print_settings import yum
from googlecloudsdk.core import config
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import creds
from googlecloudsdk.core.credentials import exceptions as creds_exceptions
from googlecloudsdk.core.credentials import store
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
def GetPythonSettingsSnippet(args):
    """Forms a Python snippet for .pypirc file (twine) and pip.conf file.

  Args:
    args: an argparse namespace. All the arguments that were provided to this
      command invocation.

  Returns:
    A python snippet.
  """
    messages = ar_requests.GetMessages()
    location, repo_path = _GetLocationAndRepoPath(args, messages.Repository.FormatValueValuesEnum.PYTHON)
    repo = _GetRequiredRepoValue(args)
    data = {'location': location, 'repo_path': repo_path, 'repo': repo}
    sa_creds = _GetServiceAccountCreds(args)
    if sa_creds:
        data['password'] = sa_creds
        return python.SERVICE_ACCOUNT_SETTING_TEMPLATE.format(**data)
    else:
        return python.NO_SERVICE_ACCOUNT_SETTING_TEMPLATE.format(**data)