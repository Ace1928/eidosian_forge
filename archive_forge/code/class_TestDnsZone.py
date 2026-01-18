from openstack.dns.v2 import _proxy
from openstack.dns.v2 import floating_ip
from openstack.dns.v2 import recordset
from openstack.dns.v2 import zone
from openstack.dns.v2 import zone_export
from openstack.dns.v2 import zone_import
from openstack.dns.v2 import zone_share
from openstack.dns.v2 import zone_transfer
from openstack.tests.unit import test_proxy_base
class TestDnsZone(TestDnsProxy):

    def test_zone_create(self):
        self.verify_create(self.proxy.create_zone, zone.Zone, method_kwargs={'name': 'id'}, expected_kwargs={'name': 'id', 'prepend_key': False})

    def test_zone_delete(self):
        self.verify_delete(self.proxy.delete_zone, zone.Zone, True, expected_kwargs={'ignore_missing': True, 'delete_shares': False})

    def test_zone_find(self):
        self.verify_find(self.proxy.find_zone, zone.Zone)

    def test_zone_get(self):
        self.verify_get(self.proxy.get_zone, zone.Zone)

    def test_zones(self):
        self.verify_list(self.proxy.zones, zone.Zone)

    def test_zone_update(self):
        self.verify_update(self.proxy.update_zone, zone.Zone)

    def test_zone_abandon(self):
        self._verify('openstack.dns.v2.zone.Zone.abandon', self.proxy.abandon_zone, method_args=[{'zone': 'id'}], expected_args=[self.proxy])

    def test_zone_xfr(self):
        self._verify('openstack.dns.v2.zone.Zone.xfr', self.proxy.xfr_zone, method_args=[{'zone': 'id'}], expected_args=[self.proxy])