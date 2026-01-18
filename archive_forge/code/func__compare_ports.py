from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.tests.unit import base
def _compare_ports(self, exp, real):
    self.assertDictEqual(_port.Port(**exp).to_dict(computed=False), real.to_dict(computed=False))