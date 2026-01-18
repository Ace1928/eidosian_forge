from openstack.dns.v2 import _proxy
from openstack.dns.v2 import floating_ip
from openstack.dns.v2 import recordset
from openstack.dns.v2 import zone
from openstack.dns.v2 import zone_export
from openstack.dns.v2 import zone_import
from openstack.dns.v2 import zone_share
from openstack.dns.v2 import zone_transfer
from openstack.tests.unit import test_proxy_base
class TestDnsZoneTransferAccept(TestDnsProxy):

    def test_zone_transfer_accept_get(self):
        self.verify_get(self.proxy.get_zone_transfer_accept, zone_transfer.ZoneTransferAccept)

    def test_zone_transfer_accepts(self):
        self.verify_list(self.proxy.zone_transfer_accepts, zone_transfer.ZoneTransferAccept)

    def test_zone_transfer_accept_create(self):
        self.verify_create(self.proxy.create_zone_transfer_accept, zone_transfer.ZoneTransferAccept)