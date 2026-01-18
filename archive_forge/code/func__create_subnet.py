from openstack.network.v2 import floating_ip
from openstack.network.v2 import network
from openstack.network.v2 import port
from openstack.network.v2 import router
from openstack.network.v2 import subnet
from openstack.tests.functional import base
def _create_subnet(self, name, net_id, cidr):
    self.name = name
    self.net_id = net_id
    self.cidr = cidr
    sub = self.user_cloud.network.create_subnet(name=self.name, ip_version=self.IPV4, network_id=self.net_id, cidr=self.cidr)
    assert isinstance(sub, subnet.Subnet)
    self.assertEqual(self.name, sub.name)
    return sub