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
class TestNetworkBGPVPN(TestNetworkProxy):
    NETWORK_ASSOCIATION = 'net-assoc-id' + uuid.uuid4().hex
    PORT_ASSOCIATION = 'port-assoc-id' + uuid.uuid4().hex
    ROUTER_ASSOCIATION = 'router-assoc-id' + uuid.uuid4().hex

    def test_bgpvpn_create(self):
        self.verify_create(self.proxy.create_bgpvpn, bgpvpn.BgpVpn)

    def test_bgpvpn_delete(self):
        self.verify_delete(self.proxy.delete_bgpvpn, bgpvpn.BgpVpn, False)

    def test_bgpvpn_delete_ignore(self):
        self.verify_delete(self.proxy.delete_bgpvpn, bgpvpn.BgpVpn, True)

    def test_bgpvpn_find(self):
        self.verify_find(self.proxy.find_bgpvpn, bgpvpn.BgpVpn)

    def test_bgpvpn_get(self):
        self.verify_get(self.proxy.get_bgpvpn, bgpvpn.BgpVpn)

    def test_bgpvpns(self):
        self.verify_list(self.proxy.bgpvpns, bgpvpn.BgpVpn)

    def test_bgpvpn_update(self):
        self.verify_update(self.proxy.update_bgpvpn, bgpvpn.BgpVpn)

    def test_bgpvpn_network_association_create(self):
        self.verify_create(self.proxy.create_bgpvpn_network_association, bgpvpn_network_association.BgpVpnNetworkAssociation, method_kwargs={'bgpvpn': BGPVPN_ID}, expected_kwargs={'bgpvpn_id': BGPVPN_ID})

    def test_bgpvpn_network_association_delete(self):
        self.verify_delete(self.proxy.delete_bgpvpn_network_association, bgpvpn_network_association.BgpVpnNetworkAssociation, False, method_args=[BGPVPN_ID, self.NETWORK_ASSOCIATION], expected_args=[self.NETWORK_ASSOCIATION], expected_kwargs={'ignore_missing': False, 'bgpvpn_id': BGPVPN_ID})

    def test_bgpvpn_network_association_delete_ignore(self):
        self.verify_delete(self.proxy.delete_bgpvpn_network_association, bgpvpn_network_association.BgpVpnNetworkAssociation, True, method_args=[BGPVPN_ID, self.NETWORK_ASSOCIATION], expected_args=[self.NETWORK_ASSOCIATION], expected_kwargs={'ignore_missing': True, 'bgpvpn_id': BGPVPN_ID})

    def test_bgpvpn_network_association_get(self):
        self.verify_get(self.proxy.get_bgpvpn_network_association, bgpvpn_network_association.BgpVpnNetworkAssociation, method_args=[BGPVPN_ID, self.NETWORK_ASSOCIATION], expected_args=[self.NETWORK_ASSOCIATION], expected_kwargs={'bgpvpn_id': BGPVPN_ID})

    def test_bgpvpn_network_associations(self):
        self.verify_list(self.proxy.bgpvpn_network_associations, bgpvpn_network_association.BgpVpnNetworkAssociation, method_args=[BGPVPN_ID], expected_args=[], expected_kwargs={'bgpvpn_id': BGPVPN_ID})

    def test_bgpvpn_port_association_create(self):
        self.verify_create(self.proxy.create_bgpvpn_port_association, bgpvpn_port_association.BgpVpnPortAssociation, method_kwargs={'bgpvpn': BGPVPN_ID}, expected_kwargs={'bgpvpn_id': BGPVPN_ID})

    def test_bgpvpn_port_association_delete(self):
        self.verify_delete(self.proxy.delete_bgpvpn_port_association, bgpvpn_port_association.BgpVpnPortAssociation, False, method_args=[BGPVPN_ID, self.PORT_ASSOCIATION], expected_args=[self.PORT_ASSOCIATION], expected_kwargs={'ignore_missing': False, 'bgpvpn_id': BGPVPN_ID})

    def test_bgpvpn_port_association_delete_ignore(self):
        self.verify_delete(self.proxy.delete_bgpvpn_port_association, bgpvpn_port_association.BgpVpnPortAssociation, True, method_args=[BGPVPN_ID, self.PORT_ASSOCIATION], expected_args=[self.PORT_ASSOCIATION], expected_kwargs={'ignore_missing': True, 'bgpvpn_id': BGPVPN_ID})

    def test_bgpvpn_port_association_find(self):
        self.verify_find(self.proxy.find_bgpvpn_port_association, bgpvpn_port_association.BgpVpnPortAssociation, method_args=[BGPVPN_ID], expected_args=['resource_name'], method_kwargs={'ignore_missing': True}, expected_kwargs={'ignore_missing': True, 'bgpvpn_id': BGPVPN_ID})

    def test_bgpvpn_port_association_get(self):
        self.verify_get(self.proxy.get_bgpvpn_port_association, bgpvpn_port_association.BgpVpnPortAssociation, method_args=[BGPVPN_ID, self.PORT_ASSOCIATION], expected_args=[self.PORT_ASSOCIATION], expected_kwargs={'bgpvpn_id': BGPVPN_ID})

    def test_bgpvpn_port_associations(self):
        self.verify_list(self.proxy.bgpvpn_port_associations, bgpvpn_port_association.BgpVpnPortAssociation, method_args=[BGPVPN_ID], expected_args=[], expected_kwargs={'bgpvpn_id': BGPVPN_ID})

    def test_bgpvpn_port_association_update(self):
        self.verify_update(self.proxy.update_bgpvpn_port_association, bgpvpn_port_association.BgpVpnPortAssociation, method_args=[BGPVPN_ID, self.PORT_ASSOCIATION], method_kwargs={}, expected_args=[self.PORT_ASSOCIATION], expected_kwargs={'bgpvpn_id': BGPVPN_ID})

    def test_bgpvpn_router_association_create(self):
        self.verify_create(self.proxy.create_bgpvpn_router_association, bgpvpn_router_association.BgpVpnRouterAssociation, method_kwargs={'bgpvpn': BGPVPN_ID}, expected_kwargs={'bgpvpn_id': BGPVPN_ID})

    def test_bgpvpn_router_association_delete(self):
        self.verify_delete(self.proxy.delete_bgpvpn_router_association, bgpvpn_router_association.BgpVpnRouterAssociation, False, method_args=[BGPVPN_ID, self.ROUTER_ASSOCIATION], expected_args=[self.ROUTER_ASSOCIATION], expected_kwargs={'ignore_missing': False, 'bgpvpn_id': BGPVPN_ID})

    def test_bgpvpn_router_association_delete_ignore(self):
        self.verify_delete(self.proxy.delete_bgpvpn_router_association, bgpvpn_router_association.BgpVpnRouterAssociation, True, method_args=[BGPVPN_ID, self.ROUTER_ASSOCIATION], expected_args=[self.ROUTER_ASSOCIATION], expected_kwargs={'ignore_missing': True, 'bgpvpn_id': BGPVPN_ID})

    def test_bgpvpn_router_association_get(self):
        self.verify_get(self.proxy.get_bgpvpn_router_association, bgpvpn_router_association.BgpVpnRouterAssociation, method_args=[BGPVPN_ID, self.ROUTER_ASSOCIATION], expected_args=[self.ROUTER_ASSOCIATION], expected_kwargs={'bgpvpn_id': BGPVPN_ID})

    def test_bgpvpn_router_associations(self):
        self.verify_list(self.proxy.bgpvpn_router_associations, bgpvpn_router_association.BgpVpnRouterAssociation, method_args=[BGPVPN_ID], expected_args=[], expected_kwargs={'bgpvpn_id': BGPVPN_ID})

    def test_bgpvpn_router_association_update(self):
        self.verify_update(self.proxy.update_bgpvpn_router_association, bgpvpn_router_association.BgpVpnRouterAssociation, method_args=[BGPVPN_ID, self.ROUTER_ASSOCIATION], method_kwargs={}, expected_args=[self.ROUTER_ASSOCIATION], expected_kwargs={'bgpvpn_id': BGPVPN_ID})