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
def _RunConnectivityTest(self):
    connectivity_test = self._CreateConnectivityTest()
    connectivity_test_create_req = self.nm_message.NetworkmanagementProjectsLocationsGlobalConnectivityTestsCreateRequest(parent='projects/{project_id}/locations/global'.format(project_id=self.project.name), testId=self.test_id, connectivityTest=connectivity_test)
    return self.nm_client.projects_locations_global_connectivityTests.Create(connectivity_test_create_req).name