import ast
import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as tempest_lib_exc
import time
from manilaclient import config
from manilaclient import exceptions
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
class ShareNetworkSecurityServiceCheckReadWriteTests(base.BaseTestCase):
    protocol = None

    def setUp(self):
        super(ShareNetworkSecurityServiceCheckReadWriteTests, self).setUp()
        if self.protocol not in CONF.enable_protocols:
            message = '%s tests are disabled.' % self.protocol
            raise self.skipException(message)
        self.client = self.get_user_client()
        if not self.client.share_network:
            message = 'Can run only with DHSS=True mode'
            raise self.skipException(message)

    def _wait_for_update_security_service_compatible_result(self, share_network, current_security_service, new_security_service=None):
        compatible_expected_result = 'True'
        check_is_compatible = 'None'
        tentatives = 0
        while check_is_compatible != compatible_expected_result:
            tentatives += 1
            if not new_security_service:
                check_is_compatible = self.user_client.share_network_security_service_add_check(share_network['id'], current_security_service['id'])['compatible']
            else:
                check_is_compatible = self.user_client.share_network_security_service_update_check(share_network['id'], current_security_service['id'], new_security_service['id'])['compatible']
            if tentatives > 3:
                timeout_message = "Share network security service add/update check did not reach 'compatible=True' within 15 seconds."
                raise exceptions.TimeoutException(message=timeout_message)
            time.sleep(5)

    def test_check_if_security_service_can_be_added_to_share_network_in_use(self):
        share_network = self.create_share_network(client=self.user_client, description='fakedescription', neutron_net_id='fake_neutron_net_id', neutron_subnet_id='fake_neutron_subnet_id')
        self.create_share(self.protocol, client=self.user_client, share_network=share_network['id'])
        current_security_service = self.create_security_service(client=self.user_client)
        check_result = self.user_client.share_network_security_service_add_check(share_network['id'], current_security_service['id'])
        self.assertEqual(check_result['compatible'], 'None')
        self._wait_for_update_security_service_compatible_result(share_network, current_security_service)

    def test_add_and_update_security_service_when_share_network_is_in_use(self):
        share_network = self.create_share_network(client=self.user_client, name='cool_net_name', description='fakedescription', neutron_net_id='fake_neutron_net_id', neutron_subnet_id='fake_neutron_subnet_id')
        self.create_share(self.protocol, name='fake_share_name', share_network=share_network['id'], client=self.user_client)
        current_security_service = self.create_security_service(client=self.user_client, name='current_security_service')
        new_security_service = self.create_security_service(client=self.user_client, name='new_security_service')
        check_result = self.user_client.share_network_security_service_add_check(share_network['id'], current_security_service['id'])
        self.assertEqual(check_result['compatible'], 'None')
        self._wait_for_update_security_service_compatible_result(share_network, current_security_service)
        self.user_client.share_network_security_service_add(share_network['id'], current_security_service['id'])
        network_services = self.user_client.share_network_security_service_list(share_network['id'])
        self.assertEqual(len(network_services), 1)
        self.assertEqual(network_services[0]['name'], current_security_service['name'])
        self.user_client.wait_for_resource_status(share_network['id'], 'active', microversion=SECURITY_SERVICE_UPDATE_VERSION, resource_type='share_network')
        check_result = self.user_client.share_network_security_service_update_check(share_network['id'], current_security_service['id'], new_security_service['id'])
        self.assertEqual(check_result['compatible'], 'None')
        self._wait_for_update_security_service_compatible_result(share_network, current_security_service, new_security_service=new_security_service)
        self.user_client.share_network_security_service_update(share_network['id'], current_security_service['id'], new_security_service['id'])
        network_services = self.user_client.share_network_security_service_list(share_network['id'])
        self.assertEqual(len(network_services), 1)
        self.assertEqual(network_services[0]['name'], new_security_service['name'])
        self.user_client.wait_for_resource_status(share_network['id'], 'active', microversion=SECURITY_SERVICE_UPDATE_VERSION, resource_type='share_network')