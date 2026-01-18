from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.compute import ssh_troubleshooter
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import log
class UserPermissionTroubleshooter(ssh_troubleshooter.SshTroubleshooter):
    """Check user permission.

  Perform IAM authorization checks for the following IAM resources: instance,
  project, service account, IAP, and OS Login if applicable.

  Attributes:
    project: The project object.
    instance: The instance object.
    zone: str, the zone name.
    iap_tunnel_args: SshTunnelArgs or None if IAP Tunnel is disabled.
  """

    def __init__(self, project, zone, instance, iap_tunnel_args):
        self.project = project
        self.zone = zone
        self.instance = instance
        self.iap_tunnel_args = iap_tunnel_args
        self.compute_client = apis.GetClientInstance(_API_COMPUTE_CLIENT_NAME, _API_CLIENT_VERSION_V1)
        self.compute_message = apis.GetMessagesModule(_API_COMPUTE_CLIENT_NAME, _API_CLIENT_VERSION_V1)
        self.iam_client = apis.GetClientInstance(_API_IAM_CLIENT_NAME, _API_CLIENT_VERSION_V1)
        self.iam_message = apis.GetMessagesModule(_API_IAM_CLIENT_NAME, _API_CLIENT_VERSION_V1)
        self.resourcemanager_client_v3 = apis.GetClientInstance(_API_RESOURCEMANAGER_CLIENT_NAME, _API_CLIENT_VERSION_V3)
        self.resourcemanager_message_v3 = apis.GetMessagesModule(_API_RESOURCEMANAGER_CLIENT_NAME, _API_CLIENT_VERSION_V3)
        self.iap_client = apis.GetClientInstance(_API_IAP_CLIENT_NAME, _API_CLIENT_VERSION_V1)
        self.iap_message = apis.GetMessagesModule(_API_IAP_CLIENT_NAME, _API_CLIENT_VERSION_V1)
        self.enable_oslogin = False
        self.issues = {}

    def check_prerequisite(self):
        """Validate if the user has enabled oslogin."""
        self.enable_oslogin = self._IsOsLoginEnabled()

    def cleanup_resources(self):
        return

    def troubleshoot(self):
        log.status.Print('---- Checking user permissions ----')
        if self.enable_oslogin:
            if self._CheckOsLoginPermissions():
                self.issues['oslogin'] = OS_LOGIN_MESSAGE
        else:
            instance_permissions.append('compute.instances.setMetadata')
            project_permissions.append('compute.projects.setCommonInstanceMetadata')
        missing_instance_project = sorted(self._CheckInstancePermissions().union(self._CheckProjectPermissions()))
        if missing_instance_project:
            self.issues['instance_project'] = INSTANCE_PROJECT_MESSAGE.format(' '.join(missing_instance_project))
        if self.instance.serviceAccounts and self._CheckServiceAccountPermissions():
            self.issues['serviceaccount'] = SERVICE_ACCOUNT_MESSAGE
        if self.iap_tunnel_args and self._CheckIapPermissions():
            self.issues['iap'] = IAP_MESSAGE
        log.status.Print('User permissions: {0} issue(s) found.\n'.format(len(self.issues.keys())))
        for message in self.issues.values():
            log.status.Print(message)

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

    def _CheckServiceAccountPermissions(self):
        """Check whether user has service account IAM permissions.

    Returns:
       set, missing IAM permissions.
    """
        iam_request = self.iam_message.TestIamPermissionsRequest(permissions=serviceaccount_permissions)
        request = self.iam_message.IamProjectsServiceAccountsTestIamPermissionsRequest(resource='projects/{project}/serviceAccounts/{serviceaccount}'.format(project=self.project.name, serviceaccount=self.instance.serviceAccounts[0].email), testIamPermissionsRequest=iam_request)
        response = self.iam_client.projects_serviceAccounts.TestIamPermissions(request)
        return set(serviceaccount_permissions) - set(response.permissions)

    def _CheckOsLoginPermissions(self):
        """Check whether user has oslogin IAM permissions.

    Returns:
      set, missing IAM permissions.
    """
        response = self._ComputeTestIamPermissions(oslogin_permissions)
        return set(oslogin_permissions) - set(response.permissions)

    def _CheckInstancePermissions(self):
        """Check whether user has IAM permission on instance resource.

    Returns:
      set, missing IAM permissions.
    """
        response = self._ComputeTestIamPermissions(instance_permissions)
        return set(instance_permissions) - set(response.permissions)

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

    def _CheckProjectPermissions(self):
        """Check whether user has IAM permission on project resource.

    Returns:
      set, missing IAM permissions.
    """
        response = self._ResourceManagerTestIamPermissions(project_permissions)
        return set(project_permissions) - set(response.permissions)

    def _ResourceManagerTestIamPermissions(self, permissions):
        """Check whether user has IAM permission on resource manager.

    Args:
      permissions: list, the permissions to check for the project resource.

    Returns:
      set, missing IAM permissions.
    """
        iam_request = self.resourcemanager_message_v3.TestIamPermissionsRequest(permissions=permissions)
        request = self.resourcemanager_message_v3.CloudresourcemanagerProjectsTestIamPermissionsRequest(resource='projects/{project}'.format(project=self.project.name), testIamPermissionsRequest=iam_request)
        return self.resourcemanager_client_v3.projects.TestIamPermissions(request)

    def _IsOsLoginEnabled(self):
        """Check whether OS Login is enabled on the VM.

    Returns:
      boolean, indicates whether OS Login is enabled.
    """
        oslogin_enabled = ssh.FeatureEnabledInMetadata(self.instance, self.project, ssh.OSLOGIN_ENABLE_METADATA_KEY)
        return bool(oslogin_enabled)