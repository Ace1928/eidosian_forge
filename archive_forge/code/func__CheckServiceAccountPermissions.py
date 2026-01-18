from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.compute import ssh_troubleshooter
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import log
def _CheckServiceAccountPermissions(self):
    """Check whether user has service account IAM permissions.

    Returns:
       set, missing IAM permissions.
    """
    iam_request = self.iam_message.TestIamPermissionsRequest(permissions=serviceaccount_permissions)
    request = self.iam_message.IamProjectsServiceAccountsTestIamPermissionsRequest(resource='projects/{project}/serviceAccounts/{serviceaccount}'.format(project=self.project.name, serviceaccount=self.instance.serviceAccounts[0].email), testIamPermissionsRequest=iam_request)
    response = self.iam_client.projects_serviceAccounts.TestIamPermissions(request)
    return set(serviceaccount_permissions) - set(response.permissions)