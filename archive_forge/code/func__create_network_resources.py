import queue
from openstack.tests.functional import base
def _create_network_resources(self):
    conn = self.conn
    self.net = conn.network.create_network(name=self.network_name)
    self.subnet = conn.network.create_subnet(name=self.getUniqueString('subnet'), network_id=self.net.id, cidr='192.169.1.0/24', ip_version=4)
    self.router = conn.network.create_router(name=self.getUniqueString('router'))
    conn.network.add_interface_to_router(self.router.id, subnet_id=self.subnet.id)