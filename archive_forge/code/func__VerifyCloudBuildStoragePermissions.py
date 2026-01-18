from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import random
import string
import time
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py.exceptions import HttpError
from apitools.base.py.exceptions import HttpNotFoundError
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.cloudbuild import logs as cb_logs
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.services import enable_api as services_api
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions as http_exc
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.artifacts import docker_util
from googlecloudsdk.command_lib.cloudbuild import execution
from googlecloudsdk.command_lib.compute.sole_tenancy import util as sole_tenancy_util
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import requests
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding as encoding_util
import six
def _VerifyCloudBuildStoragePermissions(project_id, account, applied_roles, required_storage_permissions):
    """Check for IAM permissions for an account and prompt to add if missing.

  Args:
    project_id: A string with the id of the project.
    account: A string with the identifier of an account.
    applied_roles: A set of strings containing the current roles for the
      account.
    required_storage_permissions: A set of strings containing the required
      storage permissions for the account. If a permissions isn't found, then
      the user is prompted to add these permissions in a custom role manually or
      accept adding the storage administrator role automatically.
  """
    try:
        missing_storage_permission = _FindMissingStoragePermissions(applied_roles, required_storage_permissions)
    except apitools_exceptions.HttpForbiddenError:
        missing_storage_permission = required_storage_permissions
    if not missing_storage_permission:
        return
    storage_admin_role = ROLE_STORAGE_ADMIN
    ep_table = ['{0} {1}'.format(permission, account) for permission in sorted(missing_storage_permission)]
    prompt_message = 'The following IAM permissions are needed for this operation:\n[{0}]\n'.format('\n'.join(ep_table))
    add_storage_admin = console_io.PromptContinue(message=prompt_message, prompt_string='You can add the cloud build service account to a custom role with these permissions or to the predefined role: {0}. Would you like to add it to {0}'.format(storage_admin_role), throw_if_unattended=True)
    if not add_storage_admin:
        return
    log.info('Adding [{0}] to [{1}]'.format(account, storage_admin_role))
    try:
        projects_api.AddIamPolicyBinding(project_id, account, storage_admin_role)
    except apitools_exceptions.HttpForbiddenError:
        log.warning('Your account does not have permission to add roles to the service account {0}. If import fails, ensure "{0}" has the roles "{1}" before retrying.'.format(account, storage_admin_role))
        return