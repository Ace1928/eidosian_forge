from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.functions import api_enablement
from googlecloudsdk.api_lib.functions import cmek_util
from googlecloudsdk.api_lib.functions import secrets as secrets_util
from googlecloudsdk.api_lib.functions.v1 import env_vars as env_vars_api_util
from googlecloudsdk.api_lib.functions.v1 import exceptions as function_exceptions
from googlecloudsdk.api_lib.functions.v1 import util as api_util
from googlecloudsdk.api_lib.functions.v2 import client as v2_client
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.calliope.arg_parsers import ArgumentTypeError
from googlecloudsdk.command_lib.functions import flags
from googlecloudsdk.command_lib.functions import secrets_config
from googlecloudsdk.command_lib.functions.v1.deploy import enum_util
from googlecloudsdk.command_lib.functions.v1.deploy import labels_util
from googlecloudsdk.command_lib.functions.v1.deploy import source_util
from googlecloudsdk.command_lib.functions.v1.deploy import trigger_util
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from six.moves import urllib
def _ApplySecretsArgsToFunction(function, args):
    """Populates cloud function message with secrets payload if applicable.

  It compares the CLI args with the existing secrets configuration to compute
  the effective secrets configuration.

  Args:
    function: Cloud function message to be checked and populated.
    args: All CLI arguments.

  Returns:
    updated_fields: update mask containing the list of fields to be updated.
  """
    if not secrets_config.IsArgsSpecified(args):
        return []
    old_secrets = secrets_util.GetSecretsAsDict(function.secretEnvironmentVariables, function.secretVolumes)
    new_secrets = {}
    try:
        new_secrets = secrets_config.ApplyFlags(old_secrets, args, _GetProject(), project_util.GetProjectNumber(_GetProject()))
    except ArgumentTypeError as error:
        exceptions.reraise(function_exceptions.FunctionsError(error))
    if new_secrets:
        _LogSecretsPermissionMessage(_GetProject(), function.serviceAccountEmail)
    old_secret_env_vars, old_secret_volumes = secrets_config.SplitSecretsDict(old_secrets)
    new_secret_env_vars, new_secret_volumes = secrets_config.SplitSecretsDict(new_secrets)
    updated_fields = []
    if old_secret_env_vars != new_secret_env_vars:
        function.secretEnvironmentVariables = secrets_util.SecretEnvVarsToMessages(new_secret_env_vars, api_util.GetApiMessagesModule())
        updated_fields.append('secretEnvironmentVariables')
    if old_secret_volumes != new_secret_volumes:
        function.secretVolumes = secrets_util.SecretVolumesToMessages(new_secret_volumes, api_util.GetApiMessagesModule())
        updated_fields.append('secretVolumes')
    return updated_fields