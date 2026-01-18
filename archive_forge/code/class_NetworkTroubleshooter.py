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
class NetworkTroubleshooter(ssh_troubleshooter.SshTroubleshooter):
    """Check network and firewall setting by running network connectivity test."""

    def __init__(self, project, zone, instance):
        self.project = project
        self.zone = zone
        self.instance = instance
        self.nm_client = apis.GetClientInstance(_API_NETWORKMANAGEMENT_CLIENT_NAME, _API_CLIENT_VERSION_V1)
        self.nm_message = apis.GetMessagesModule(_API_NETWORKMANAGEMENT_CLIENT_NAME, _API_CLIENT_VERSION_V1)
        self.compute_client = apis.GetClientInstance(_API_COMPUTE_CLIENT_NAME, _API_CLIENT_VERSION_V1)
        self.compute_message = apis.GetMessagesModule(_API_COMPUTE_CLIENT_NAME, _API_CLIENT_VERSION_V1)
        self.skip_troubleshoot = False
        self.test_id = 'ssh-troubleshoot-' + _GetRandomSuffix()

    def check_prerequisite(self):
        log.status.Print('---- Checking network connectivity ----')
        msg = "The Network Management API is needed to check the VM's network connectivity."
        prompt = "Is it OK to enable it and check the VM's network connectivity?"
        cancel = 'Test skipped.\nTo manually test network connectivity, try reaching another device on the same network.\n'
        try:
            prompt_continue = console_io.PromptContinue(message=msg, prompt_string=prompt, cancel_on_no=True, cancel_string=cancel)
            self.skip_troubleshoot = not prompt_continue
        except OperationCancelledError:
            self.skip_troubleshoot = True
        if self.skip_troubleshoot:
            return
        enable_api.EnableService(self.project.name, NETWORK_API)
        missing_permissions = self._CheckNetworkManagementPermissions()
        if missing_permissions:
            log.status.Print('Missing the IAM permissions {0} necessary to perform the network connectivity test. To manually test network connectivity, try reaching another device on the same network.\n'.format(' '.join(missing_permissions)))
            self.skip_troubleshoot = True
            return

    def cleanup_resources(self):
        return

    def troubleshoot(self):
        if self.skip_troubleshoot:
            return
        self.ip_address = self._GetSourceIPAddress()
        log.status.Print('Your source IP address is {0}\n'.format(self.ip_address))
        if not self.ip_address:
            log.status.Print("Could not resolve source external IP address, can't run network connectivity test.\n")
            self.skip_troubleshoot = True
            return
        operation_name = self._RunConnectivityTest()
        while not self._IsConnectivityTestFinish(operation_name):
            time.sleep(1)
        test_result = self._GetConnectivityTestResult()
        self._PrintConciseConnectivityTestResult(test_result)
        log.status.Print(CONNECTIVITY_TEST_MESSAGE.format(self.test_id, self.project.name))
        return

    def _RunConnectivityTest(self):
        connectivity_test = self._CreateConnectivityTest()
        connectivity_test_create_req = self.nm_message.NetworkmanagementProjectsLocationsGlobalConnectivityTestsCreateRequest(parent='projects/{project_id}/locations/global'.format(project_id=self.project.name), testId=self.test_id, connectivityTest=connectivity_test)
        return self.nm_client.projects_locations_global_connectivityTests.Create(connectivity_test_create_req).name

    def _GetConnectivityTestResult(self):
        name = 'projects/{project_id}/locations/global/connectivityTests/{test_id}'.format(project_id=self.project.name, test_id=self.test_id)
        connectivity_test_get_req = self.nm_message.NetworkmanagementProjectsLocationsGlobalConnectivityTestsGetRequest(name=name)
        return self.nm_client.projects_locations_global_connectivityTests.Get(connectivity_test_get_req)

    def _IsConnectivityTestFinish(self, name):
        operation_get_req = self.nm_message.NetworkmanagementProjectsLocationsGlobalOperationsGetRequest(name=name)
        return self.nm_client.projects_locations_global_operations.Get(operation_get_req).done

    def _CreateConnectivityTest(self):
        return self.nm_message.ConnectivityTest(name='projects/{name}/locations/global/connectivityTests/{testId}'.format(name=self.project.name, testId=self.test_id), description="This connectivity test is created by 'gcloud compute ssh --troubleshoot'", source=self.nm_message.Endpoint(ipAddress=self.ip_address, projectId=self.project.name), destination=self.nm_message.Endpoint(port=22, instance='projects/{project}/zones/{zone}/instances/{instance}'.format(project=self.project.name, zone=self.zone, instance=self.instance.name)), protocol='TCP')

    def _CheckNetworkManagementPermissions(self):
        resource_url = 'projects/{project_id}/locations/global/connectivityTests/*'.format(project_id=self.project.name)
        test_permission_req = self.nm_message.TestIamPermissionsRequest(permissions=networkmanagement_permissions)
        nm_testiampermission_req = self.nm_message.NetworkmanagementProjectsLocationsGlobalConnectivityTestsTestIamPermissionsRequest(resource=resource_url, testIamPermissionsRequest=test_permission_req)
        response = self.nm_client.projects_locations_global_connectivityTests.TestIamPermissions(nm_testiampermission_req)
        return set(networkmanagement_permissions) - set(response.permissions)

    def _GetSourceIPAddress(self):
        """Get current external IP from Google DNS server.

    Returns:
      str, an ipv4 address represented by string
    """
        re = resolver.Resolver()
        re.nameservers = [socket.gethostbyname('ns1.google.com')]
        for rdata in re.query(qname='o-o.myaddr.l.google.com', rdtype='TXT'):
            return six.text_type(rdata).strip('"')
        return ''

    def _PrintConciseConnectivityTestResult(self, response):
        """Print concise network connectivity test result from response.

    Args:
      response: A response from projects_locations_global_connectivityTests Get

    Returns:

    """
        details = response.reachabilityDetails
        if details:
            log.status.Print('Network Connectivity Test Result: {0}\n'.format(details.result))