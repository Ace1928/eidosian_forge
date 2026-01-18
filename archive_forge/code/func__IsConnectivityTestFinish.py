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
def _IsConnectivityTestFinish(self, name):
    operation_get_req = self.nm_message.NetworkmanagementProjectsLocationsGlobalOperationsGetRequest(name=name)
    return self.nm_client.projects_locations_global_operations.Get(operation_get_req).done