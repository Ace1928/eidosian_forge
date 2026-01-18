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
def GetNpmSettingsSnippet(args):
    """Forms an npm settings snippet to add to the .npmrc file.

  Args:
    args: an argparse namespace. All the arguments that were provided to this
      command invocation.

  Returns:
    An npm settings snippet.
  """
    messages = ar_requests.GetMessages()
    location, repo_path = _GetLocationAndRepoPath(args, messages.Repository.FormatValueValuesEnum.NPM)
    registry_path = '{location}-npm.pkg.dev/{repo_path}/'.format(**{'location': location, 'repo_path': repo_path})
    configured_registry = 'registry'
    if args.scope:
        if not args.scope.startswith('@') or len(args.scope) <= 1:
            raise ar_exceptions.InvalidInputValueError('Scope name must start with "@" and be longer than 1 character.')
        configured_registry = args.scope + ':' + configured_registry
    data = {'configured_registry': configured_registry, 'registry_path': registry_path, 'repo_path': repo_path}
    sa_creds = _GetServiceAccountCreds(args)
    if sa_creds:
        npm_setting_template = npm.SERVICE_ACCOUNT_TEMPLATE
        data['password'] = base64.b64encode(sa_creds.encode('utf-8')).decode('utf-8')
    else:
        npm_setting_template = npm.NO_SERVICE_ACCOUNT_TEMPLATE
    return npm_setting_template.format(**data)