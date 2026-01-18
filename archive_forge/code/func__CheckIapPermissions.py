from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.compute import ssh_troubleshooter
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import log
def _CheckIapPermissions(self):
    """Check if user miss any IAP Permissions.

    Returns:
      set, missing IAM permissions.
    """
    iam_request = self.iap_message.TestIamPermissionsRequest(permissions=iap_permissions)
    resource = 'projects/{}/iap_tunnel/zones/{}/instances/{}'.format(self.project.name, self.zone, self.instance.name)
    request = self.iap_message.IapTestIamPermissionsRequest(resource=resource, testIamPermissionsRequest=iam_request)
    response = self.iap_client.v1.TestIamPermissions(request)
    return set(iap_permissions) - set(response.permissions)