from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import random
import socket
import string
import time
from dns import resolver
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.compute import ssh_troubleshooter
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console.console_io import OperationCancelledError
import six
def _CheckNetworkManagementPermissions(self):
    resource_url = 'projects/{project_id}/locations/global/connectivityTests/*'.format(project_id=self.project.name)
    test_permission_req = self.nm_message.TestIamPermissionsRequest(permissions=networkmanagement_permissions)
    nm_testiampermission_req = self.nm_message.NetworkmanagementProjectsLocationsGlobalConnectivityTestsTestIamPermissionsRequest(resource=resource_url, testIamPermissionsRequest=test_permission_req)
    response = self.nm_client.projects_locations_global_connectivityTests.TestIamPermissions(nm_testiampermission_req)
    return set(networkmanagement_permissions) - set(response.permissions)