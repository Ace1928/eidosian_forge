from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.iam import iam_util
def CreateServiceAccount(self, project, account_id):
    """Creates a service account within the provided project.

    Args:
      project: The project string to create a service account within.
      account_id: The string id to create the service account with.

    Returns:
      A string email of the service account.
    """
    messages = self._iam_client.MESSAGES_MODULE
    response = self._iam_client.projects_serviceAccounts.Create(messages.IamProjectsServiceAccountsCreateRequest(name=iam_util.ProjectToProjectResourceName(project), createServiceAccountRequest=messages.CreateServiceAccountRequest(accountId=account_id)))
    return response.email