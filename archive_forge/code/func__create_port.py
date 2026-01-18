from openstack.network.v2 import network as _network
from openstack.network.v2 import port as _port
from openstack.network.v2 import sfc_flow_classifier as _flow_classifier
from openstack.network.v2 import subnet as _subnet
from openstack.tests.functional import base
def _create_port(self, network, port_name):
    port = self.op_net_client.create_port(name=port_name, network_id=network.id)
    self.assertIsInstance(port, _port.Port)
    return port