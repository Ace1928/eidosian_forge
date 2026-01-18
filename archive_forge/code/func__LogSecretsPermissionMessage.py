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
def _LogSecretsPermissionMessage(project, service_account_email):
    """Logs a warning message asking the user to grant access to secrets.

  This will be removed once access checker is added.

  Args:
    project: Project id.
    service_account_email: Runtime service account email.
  """
    if not service_account_email:
        service_account_email = '{project}@appspot.gserviceaccount.com'.format(project=project)
    message = "This deployment uses secrets. Ensure that the runtime service account '{sa}' has access to the secrets. You can do that by granting the permission 'roles/secretmanager.secretAccessor' to the runtime service account on the project or secrets.\n"
    command = "E.g. gcloud projects add-iam-policy-binding {project} --member='serviceAccount:{sa}' --role='roles/secretmanager.secretAccessor'"
    log.warning((message + command).format(project=project, sa=service_account_email))