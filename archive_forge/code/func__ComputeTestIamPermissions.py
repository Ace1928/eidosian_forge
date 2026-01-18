from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.compute import ssh_troubleshooter
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import log
def _ComputeTestIamPermissions(self, permissions):
    """Call TestIamPermissions to check whether user has certain IAM permissions.

    Args:
      permissions: list, the permissions to check for the instance resource.

    Returns:
      TestPermissionsResponse, the API response from TestIamPermissions.
    """
    iam_request = self.compute_message.TestPermissionsRequest(permissions=permissions)
    request = self.compute_message.ComputeInstancesTestIamPermissionsRequest(project=self.project.name, resource=self.instance.name, testPermissionsRequest=iam_request, zone=self.zone)
    return self.compute_client.instances.TestIamPermissions(request)