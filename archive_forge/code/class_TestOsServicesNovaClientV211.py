from novaclient.tests.functional import base
from novaclient.tests.functional.v2.legacy import test_os_services
from novaclient import utils
class TestOsServicesNovaClientV211(test_os_services.TestOsServicesNovaClient):
    """Functional tests for os-services attributes, microversion 2.11"""
    COMPUTE_API_VERSION = '2.11'

    def test_os_services_force_down_force_up(self):
        for serv in self.client.services.list():
            if serv.binary != 'nova-compute':
                continue
            service_list = self.nova('service-list --binary %s' % serv.binary)
            status = self._get_column_value_from_single_row_table(service_list, 'Forced down')
            self.assertEqual('False', status)
            host = self._get_column_value_from_single_row_table(service_list, 'Host')
            service = self.nova('service-force-down %s' % host)
            self.addCleanup(self.nova, 'service-force-down --unset', params=host)
            status = self._get_column_value_from_single_row_table(service, 'Forced down')
            self.assertEqual('True', status)
            service = self.nova('service-force-down --unset %s' % host)
            status = self._get_column_value_from_single_row_table(service, 'Forced down')
            self.assertEqual('False', status)