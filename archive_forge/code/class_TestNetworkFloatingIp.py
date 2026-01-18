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
class TestNetworkFloatingIp(TestNetworkProxy):

    def test_create_floating_ip_port_forwarding(self):
        self.verify_create(self.proxy.create_floating_ip_port_forwarding, port_forwarding.PortForwarding, method_kwargs={'floating_ip': FIP_ID}, expected_kwargs={'floatingip_id': FIP_ID})

    def test_delete_floating_ip_port_forwarding(self):
        self.verify_delete(self.proxy.delete_floating_ip_port_forwarding, port_forwarding.PortForwarding, ignore_missing=False, method_args=[FIP_ID, 'resource_or_id'], expected_args=['resource_or_id'], expected_kwargs={'floatingip_id': FIP_ID})

    def test_delete_floating_ip_port_forwarding_ignore(self):
        self.verify_delete(self.proxy.delete_floating_ip_port_forwarding, port_forwarding.PortForwarding, ignore_missing=True, method_args=[FIP_ID, 'resource_or_id'], expected_args=['resource_or_id'], expected_kwargs={'floatingip_id': FIP_ID})

    def test_find_floating_ip_port_forwarding(self):
        fip = floating_ip.FloatingIP.new(id=FIP_ID)
        self._verify('openstack.proxy.Proxy._find', self.proxy.find_floating_ip_port_forwarding, method_args=[fip, 'port_forwarding_id'], expected_args=[port_forwarding.PortForwarding, 'port_forwarding_id'], expected_kwargs={'ignore_missing': True, 'floatingip_id': FIP_ID})

    def test_get_floating_ip_port_forwarding(self):
        fip = floating_ip.FloatingIP.new(id=FIP_ID)
        self._verify('openstack.proxy.Proxy._get', self.proxy.get_floating_ip_port_forwarding, method_args=[fip, 'port_forwarding_id'], expected_args=[port_forwarding.PortForwarding, 'port_forwarding_id'], expected_kwargs={'floatingip_id': FIP_ID})

    def test_floating_ip_port_forwardings(self):
        self.verify_list(self.proxy.floating_ip_port_forwardings, port_forwarding.PortForwarding, method_kwargs={'floating_ip': FIP_ID}, expected_kwargs={'floatingip_id': FIP_ID})

    def test_update_floating_ip_port_forwarding(self):
        fip = floating_ip.FloatingIP.new(id=FIP_ID)
        self._verify('openstack.network.v2._proxy.Proxy._update', self.proxy.update_floating_ip_port_forwarding, method_args=[fip, 'port_forwarding_id'], method_kwargs={'foo': 'bar'}, expected_args=[port_forwarding.PortForwarding, 'port_forwarding_id'], expected_kwargs={'floatingip_id': FIP_ID, 'foo': 'bar'})

    def test_create_l3_conntrack_helper(self):
        self.verify_create(self.proxy.create_conntrack_helper, l3_conntrack_helper.ConntrackHelper, method_kwargs={'router': ROUTER_ID}, expected_kwargs={'router_id': ROUTER_ID})

    def test_delete_l3_conntrack_helper(self):
        r = router.Router.new(id=ROUTER_ID)
        self.verify_delete(self.proxy.delete_conntrack_helper, l3_conntrack_helper.ConntrackHelper, ignore_missing=False, method_args=['resource_or_id', r], expected_args=['resource_or_id'], expected_kwargs={'router_id': ROUTER_ID})

    def test_delete_l3_conntrack_helper_ignore(self):
        r = router.Router.new(id=ROUTER_ID)
        self.verify_delete(self.proxy.delete_conntrack_helper, l3_conntrack_helper.ConntrackHelper, ignore_missing=True, method_args=['resource_or_id', r], expected_args=['resource_or_id'], expected_kwargs={'router_id': ROUTER_ID})

    def test_get_l3_conntrack_helper(self):
        r = router.Router.new(id=ROUTER_ID)
        self._verify('openstack.proxy.Proxy._get', self.proxy.get_conntrack_helper, method_args=['conntrack_helper_id', r], expected_args=[l3_conntrack_helper.ConntrackHelper, 'conntrack_helper_id'], expected_kwargs={'router_id': ROUTER_ID})

    def test_l3_conntrack_helpers(self):
        self.verify_list(self.proxy.conntrack_helpers, l3_conntrack_helper.ConntrackHelper, method_args=[ROUTER_ID], expected_args=[], expected_kwargs={'router_id': ROUTER_ID})

    def test_update_l3_conntrack_helper(self):
        r = router.Router.new(id=ROUTER_ID)
        self._verify('openstack.network.v2._proxy.Proxy._update', self.proxy.update_conntrack_helper, method_args=['conntrack_helper_id', r], method_kwargs={'foo': 'bar'}, expected_args=[l3_conntrack_helper.ConntrackHelper, 'conntrack_helper_id'], expected_kwargs={'router_id': ROUTER_ID, 'foo': 'bar'})