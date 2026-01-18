from openstack.shared_file_system.v2 import share_network_subnet as SNS
from openstack.tests.unit import base
class TestShareNetworkSubnet(base.TestCase):

    def test_basic(self):
        SNS_resource = SNS.ShareNetworkSubnet()
        self.assertEqual('share_network_subnets', SNS_resource.resources_key)
        self.assertEqual('/share-networks/%(share_network_id)s/subnets', SNS_resource.base_path)
        self.assertTrue(SNS_resource.allow_list)

    def test_make_share_network_subnet(self):
        SNS_resource = SNS.ShareNetworkSubnet(**EXAMPLE)
        self.assertEqual(EXAMPLE['id'], SNS_resource.id)
        self.assertEqual(EXAMPLE['availability_zone'], SNS_resource.availability_zone)
        self.assertEqual(EXAMPLE['share_network_id'], SNS_resource.share_network_id)
        self.assertEqual(EXAMPLE['share_network_name'], SNS_resource.share_network_name)
        self.assertEqual(EXAMPLE['created_at'], SNS_resource.created_at)
        self.assertEqual(EXAMPLE['segmentation_id'], SNS_resource.segmentation_id)
        self.assertEqual(EXAMPLE['neutron_subnet_id'], SNS_resource.neutron_subnet_id)
        self.assertEqual(EXAMPLE['updated_at'], SNS_resource.updated_at)
        self.assertEqual(EXAMPLE['neutron_net_id'], SNS_resource.neutron_net_id)
        self.assertEqual(EXAMPLE['ip_version'], SNS_resource.ip_version)
        self.assertEqual(EXAMPLE['cidr'], SNS_resource.cidr)
        self.assertEqual(EXAMPLE['network_type'], SNS_resource.network_type)
        self.assertEqual(EXAMPLE['mtu'], SNS_resource.mtu)
        self.assertEqual(EXAMPLE['gateway'], SNS_resource.gateway)