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
def _CreateConnectivityTest(self):
    return self.nm_message.ConnectivityTest(name='projects/{name}/locations/global/connectivityTests/{testId}'.format(name=self.project.name, testId=self.test_id), description="This connectivity test is created by 'gcloud compute ssh --troubleshoot'", source=self.nm_message.Endpoint(ipAddress=self.ip_address, projectId=self.project.name), destination=self.nm_message.Endpoint(port=22, instance='projects/{project}/zones/{zone}/instances/{instance}'.format(project=self.project.name, zone=self.zone, instance=self.instance.name)), protocol='TCP')