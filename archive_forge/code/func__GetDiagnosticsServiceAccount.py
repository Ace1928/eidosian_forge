from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import datetime
import json
import time
from apitools.base.py.exceptions import HttpError
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.diagnose import diagnose_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.command_lib.util import time_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
import six
def _GetDiagnosticsServiceAccount(self, project):
    """Locates or creates a service account with the correct permissions.

    Attempts to locate the service account meant for creating the signed url.
    If not found, it will subsequently create the service account. It will then
    give the service account the correct IAM permissions to create a signed url
    to a GCS Bucket.

    Args:
      project: The project to search for the service account in.

    Returns:
      A string email of the service account to use.
    """
    service_account = None
    for account in self._diagnose_client.ListServiceAccounts(project):
        if account.email.startswith('{}@'.format(_SERVICE_ACCOUNT_NAME)):
            service_account = account.email
    if service_account is None:
        service_account = self._diagnose_client.CreateServiceAccount(project, _SERVICE_ACCOUNT_NAME)
    project_ref = projects_util.ParseProject(project)
    service_account_ref = 'serviceAccount:{}'.format(service_account)
    projects_api.AddIamPolicyBinding(project_ref, service_account_ref, 'roles/storage.objectCreator')
    projects_api.AddIamPolicyBinding(project_ref, service_account_ref, 'roles/storage.objectViewer')
    return service_account