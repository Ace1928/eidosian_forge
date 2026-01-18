from unittest import mock
import uuid
from openstack import exceptions
from openstack.network.v2 import _proxy
from openstack.network.v2 import address_group
from openstack.network.v2 import address_scope
from openstack.network.v2 import agent
from openstack.network.v2 import auto_allocated_topology
from openstack.network.v2 import availability_zone
from openstack.network.v2 import bgp_peer
from openstack.network.v2 import bgp_speaker
from openstack.network.v2 import bgpvpn
from openstack.network.v2 import bgpvpn_network_association
from openstack.network.v2 import bgpvpn_port_association
from openstack.network.v2 import bgpvpn_router_association
from openstack.network.v2 import extension
from openstack.network.v2 import firewall_group
from openstack.network.v2 import firewall_policy
from openstack.network.v2 import firewall_rule
from openstack.network.v2 import flavor
from openstack.network.v2 import floating_ip
from openstack.network.v2 import health_monitor
from openstack.network.v2 import l3_conntrack_helper
from openstack.network.v2 import listener
from openstack.network.v2 import load_balancer
from openstack.network.v2 import local_ip
from openstack.network.v2 import local_ip_association
from openstack.network.v2 import metering_label
from openstack.network.v2 import metering_label_rule
from openstack.network.v2 import ndp_proxy
from openstack.network.v2 import network
from openstack.network.v2 import network_ip_availability
from openstack.network.v2 import network_segment_range
from openstack.network.v2 import pool
from openstack.network.v2 import pool_member
from openstack.network.v2 import port
from openstack.network.v2 import port_forwarding
from openstack.network.v2 import qos_bandwidth_limit_rule
from openstack.network.v2 import qos_dscp_marking_rule
from openstack.network.v2 import qos_minimum_bandwidth_rule
from openstack.network.v2 import qos_minimum_packet_rate_rule
from openstack.network.v2 import qos_policy
from openstack.network.v2 import qos_rule_type
from openstack.network.v2 import quota
from openstack.network.v2 import rbac_policy
from openstack.network.v2 import router
from openstack.network.v2 import security_group
from openstack.network.v2 import security_group_rule
from openstack.network.v2 import segment
from openstack.network.v2 import service_profile
from openstack.network.v2 import service_provider
from openstack.network.v2 import subnet
from openstack.network.v2 import subnet_pool
from openstack.network.v2 import vpn_endpoint_group
from openstack.network.v2 import vpn_ike_policy
from openstack.network.v2 import vpn_ipsec_policy
from openstack.network.v2 import vpn_ipsec_site_connection
from openstack.network.v2 import vpn_service
from openstack import proxy as proxy_base
from openstack.tests.unit import test_proxy_base
class TestNetworkTags(TestNetworkProxy):

    def test_set_tags(self):
        x_network = network.Network.new(id='NETWORK_ID')
        self._verify('openstack.network.v2.network.Network.set_tags', self.proxy.set_tags, method_args=[x_network, ['TAG1', 'TAG2']], expected_args=[self.proxy, ['TAG1', 'TAG2']], expected_result=mock.sentinel.result_set_tags)

    @mock.patch('openstack.network.v2.network.Network.set_tags')
    def test_set_tags_resource_without_tag_suport(self, mock_set_tags):
        no_tag_resource = object()
        self.assertRaises(exceptions.InvalidRequest, self.proxy.set_tags, no_tag_resource, ['TAG1', 'TAG2'])
        self.assertEqual(0, mock_set_tags.call_count)