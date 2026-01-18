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
def _CheckIamPermissions(project_id, cloudbuild_service_account_roles, compute_service_account_roles, custom_cloudbuild_service_account='', custom_compute_service_account=''):
    """Check for needed IAM permissions and prompt to add if missing.

  Args:
    project_id: A string with the id of the project.
    cloudbuild_service_account_roles: A set of roles required for cloudbuild
      service account.
    compute_service_account_roles: A set of roles required for compute service
      account.
    custom_cloudbuild_service_account: Custom cloudbuild service account
    custom_compute_service_account: Custom compute service account
  """
    project = projects_api.Get(project_id)
    expected_services = ['cloudbuild.googleapis.com', 'logging.googleapis.com', 'compute.googleapis.com']
    for service_name in expected_services:
        if not services_api.IsServiceEnabled(project.projectId, service_name):
            prompt_message = 'The "{0}" service is not enabled for this project. It is required for this operation.\n'.format(service_name)
            enable_service = console_io.PromptContinue(prompt_message, 'Would you like to enable this service?', throw_if_unattended=True)
            if enable_service:
                services_api.EnableService(project.projectId, service_name)
            else:
                log.warning('If import fails, manually enable {0} before retrying. For instructions on enabling services, see https://cloud.google.com/service-usage/docs/enable-disable.'.format(service_name))
    build_account = 'serviceAccount:{0}@cloudbuild.gserviceaccount.com'.format(project.projectNumber)
    if custom_cloudbuild_service_account:
        build_account = 'serviceAccount:{0}'.format(custom_cloudbuild_service_account)
    compute_account = 'serviceAccount:{0}-compute@developer.gserviceaccount.com'.format(project.projectNumber)
    if custom_compute_service_account:
        compute_account = 'serviceAccount:{0}'.format(custom_compute_service_account)
    try:
        policy = projects_api.GetIamPolicy(project_id)
    except apitools_exceptions.HttpForbiddenError:
        log.warning('Your account does not have permission to check roles for the service account {0}. If import fails, ensure "{0}" has the roles "{1}" and "{2}" has the roles "{3}" before retrying.'.format(build_account, cloudbuild_service_account_roles, compute_account, compute_service_account_roles))
        return
    current_cloudbuild_account_roles = _CurrentRolesForAccount(policy, build_account)
    _VerifyCloudBuildStoragePermissions(project_id, build_account, current_cloudbuild_account_roles, CLOUD_BUILD_STORAGE_PERMISSIONS)
    _VerifyRolesAndPromptIfMissing(project_id, build_account, current_cloudbuild_account_roles, frozenset(cloudbuild_service_account_roles))
    current_compute_account_roles = _CurrentRolesForAccount(policy, compute_account)
    if ROLE_EDITOR not in current_compute_account_roles:
        _VerifyRolesAndPromptIfMissing(project_id, compute_account, current_compute_account_roles, compute_service_account_roles)