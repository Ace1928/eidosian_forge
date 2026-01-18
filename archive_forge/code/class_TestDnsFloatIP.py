from openstack.dns.v2 import _proxy
from openstack.dns.v2 import floating_ip
from openstack.dns.v2 import recordset
from openstack.dns.v2 import zone
from openstack.dns.v2 import zone_export
from openstack.dns.v2 import zone_import
from openstack.dns.v2 import zone_share
from openstack.dns.v2 import zone_transfer
from openstack.tests.unit import test_proxy_base
class TestDnsFloatIP(TestDnsProxy):

    def test_floating_ips(self):
        self.verify_list(self.proxy.floating_ips, floating_ip.FloatingIP)

    def test_floating_ip_get(self):
        self.verify_get(self.proxy.get_floating_ip, floating_ip.FloatingIP)

    def test_floating_ip_update(self):
        self.verify_update(self.proxy.update_floating_ip, floating_ip.FloatingIP)

    def test_floating_ip_unset(self):
        self._verify('openstack.proxy.Proxy._update', self.proxy.unset_floating_ip, method_args=['value'], method_kwargs={}, expected_args=[floating_ip.FloatingIP, 'value'], expected_kwargs={'ptrdname': None})