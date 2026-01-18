import copy
import testtools
from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.network.v2 import router as _router
from openstack.tests.unit import base
def _test_list_router_interfaces(self, router, interface_type, expected_result=None):
    internal_ports = [{'id': 'internal_port_id', 'fixed_ips': [{'subnet_id': 'internal_subnet_id', 'ip_address': '10.0.0.1'}], 'device_id': self.router_id, 'device_owner': device_owner} for device_owner in ['network:router_interface', 'network:ha_router_replicated_interface', 'network:router_interface_distributed']]
    external_ports = [{'id': 'external_port_id', 'fixed_ips': [{'subnet_id': 'external_subnet_id', 'ip_address': '1.2.3.4'}], 'device_id': self.router_id, 'device_owner': 'network:router_gateway'}]
    if expected_result is None:
        if interface_type == 'internal':
            expected_result = internal_ports
        elif interface_type == 'external':
            expected_result = external_ports
        else:
            expected_result = internal_ports + external_ports
    mock_uri = dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'ports'], qs_elements=['device_id=%s' % self.router_id]), json={'ports': internal_ports + external_ports})
    self.register_uris([mock_uri])
    ret = self.cloud.list_router_interfaces(router, interface_type)
    self.assertEqual([_port.Port(**i).to_dict(computed=False) for i in expected_result], [i.to_dict(computed=False) for i in ret])
    self.assert_calls()