from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.compute import ssh_troubleshooter
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import log
def _CheckProjectPermissions(self):
    """Check whether user has IAM permission on project resource.

    Returns:
      set, missing IAM permissions.
    """
    response = self._ResourceManagerTestIamPermissions(project_permissions)
    return set(project_permissions) - set(response.permissions)