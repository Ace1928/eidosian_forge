from openstack.dns.v2 import _proxy
from openstack.dns.v2 import floating_ip
from openstack.dns.v2 import recordset
from openstack.dns.v2 import zone
from openstack.dns.v2 import zone_export
from openstack.dns.v2 import zone_import
from openstack.dns.v2 import zone_share
from openstack.dns.v2 import zone_transfer
from openstack.tests.unit import test_proxy_base
class TestDnsZoneShare(TestDnsProxy):

    def test_zone_share_create(self):
        self.verify_create(self.proxy.create_zone_share, zone_share.ZoneShare, method_kwargs={'zone': 'bogus_id'}, expected_kwargs={'zone_id': 'bogus_id'})

    def test_zone_share_delete(self):
        self.verify_delete(self.proxy.delete_zone_share, zone_share.ZoneShare, ignore_missing=True, method_args={'zone': 'bogus_id', 'zone_share': 'bogus_id'}, expected_args=['zone_share'], expected_kwargs={'zone_id': 'zone', 'ignore_missing': True})

    def test_zone_share_find(self):
        self.verify_find(self.proxy.find_zone_share, zone_share.ZoneShare, method_args=['zone'], expected_args=['zone'], expected_kwargs={'zone_id': 'resource_name', 'ignore_missing': True})

    def test_zone_share_get(self):
        self.verify_get(self.proxy.get_zone_share, zone_share.ZoneShare, method_args=['zone', 'zone_share'], expected_args=['zone_share'], expected_kwargs={'zone_id': 'zone'})

    def test_zone_shares(self):
        self.verify_list(self.proxy.zone_shares, zone_share.ZoneShare, method_args=['zone'], expected_args=[], expected_kwargs={'zone_id': 'zone'})