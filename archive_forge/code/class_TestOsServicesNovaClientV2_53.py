from novaclient.tests.functional import base
from novaclient.tests.functional.v2.legacy import test_os_services
from novaclient import utils
class TestOsServicesNovaClientV2_53(base.ClientTestBase):
    """Tests the nova service-* commands using the 2.53 microversion.

    The main difference with the 2.53 microversion in these commands is
    the host/binary combination is replaced with the service.id as the
    unique identifier for a service.
    """
    COMPUTE_API_VERSION = '2.53'

    def test_os_services_list(self):
        table = self.nova('service-list')
        for serv in self.client.services.list():
            self.assertIn(serv.binary, table)
            self.assertFalse(utils.is_integer_like(serv.id))
            self.assertIn(serv.id, table)

    def test_os_service_disable_enable(self):
        for serv in self.client.services.list():
            if serv.binary != 'nova-compute':
                continue
            service = self.nova('service-disable %s' % serv.id)
            self.addCleanup(self.nova, 'service-enable', params='%s' % serv.id)
            service_id = self._get_column_value_from_single_row_table(service, 'ID')
            self.assertEqual(serv.id, service_id)
            status = self._get_column_value_from_single_row_table(service, 'Status')
            self.assertEqual('disabled', status)
            service = self.nova('service-enable %s' % serv.id)
            service_id = self._get_column_value_from_single_row_table(service, 'ID')
            self.assertEqual(serv.id, service_id)
            status = self._get_column_value_from_single_row_table(service, 'Status')
            self.assertEqual('enabled', status)

    def test_os_service_disable_log_reason(self):
        for serv in self.client.services.list():
            if serv.binary != 'nova-compute':
                continue
            service = self.nova('service-disable --reason test_disable %s' % serv.id)
            self.addCleanup(self.nova, 'service-enable', params='%s' % serv.id)
            service_id = self._get_column_value_from_single_row_table(service, 'ID')
            self.assertEqual(serv.id, service_id)
            status = self._get_column_value_from_single_row_table(service, 'Status')
            log_reason = self._get_column_value_from_single_row_table(service, 'Disabled Reason')
            self.assertEqual('disabled', status)
            self.assertEqual('test_disable', log_reason)

    def test_os_services_force_down_force_up(self):
        for serv in self.client.services.list():
            if serv.binary != 'nova-compute':
                continue
            service = self.nova('service-force-down %s' % serv.id)
            self.addCleanup(self.nova, 'service-force-down --unset', params='%s' % serv.id)
            service_id = self._get_column_value_from_single_row_table(service, 'ID')
            self.assertEqual(serv.id, service_id)
            forced_down = self._get_column_value_from_single_row_table(service, 'Forced down')
            self.assertEqual('True', forced_down)
            service = self.nova('service-force-down --unset %s' % serv.id)
            forced_down = self._get_column_value_from_single_row_table(service, 'Forced down')
            self.assertEqual('False', forced_down)