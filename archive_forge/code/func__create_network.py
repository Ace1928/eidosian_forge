from openstack.network.v2 import floating_ip
from openstack.network.v2 import network
from openstack.network.v2 import port
from openstack.network.v2 import router
from openstack.network.v2 import subnet
from openstack.tests.functional import base
def _create_network(self, name, **args):
    self.name = name
    net = self.user_cloud.network.create_network(name=name, **args)
    assert isinstance(net, network.Network)
    self.assertEqual(self.name, net.name)
    return net