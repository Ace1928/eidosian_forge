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
class TestNetworkQuota(TestNetworkProxy):

    def test_quota_delete(self):
        self.verify_delete(self.proxy.delete_quota, quota.Quota, False)

    def test_quota_delete_ignore(self):
        self.verify_delete(self.proxy.delete_quota, quota.Quota, True)

    def test_quota_get(self):
        self.verify_get(self.proxy.get_quota, quota.Quota)

    @mock.patch.object(proxy_base.Proxy, '_get_resource')
    def test_quota_get_details(self, mock_get):
        fake_quota = mock.Mock(project_id='PROJECT')
        mock_get.return_value = fake_quota
        self._verify('openstack.proxy.Proxy._get', self.proxy.get_quota, method_args=['QUOTA_ID'], method_kwargs={'details': True}, expected_args=[quota.QuotaDetails], expected_kwargs={'project': fake_quota.id, 'requires_id': False})
        mock_get.assert_called_once_with(quota.Quota, 'QUOTA_ID')

    @mock.patch.object(proxy_base.Proxy, '_get_resource')
    def test_quota_default_get(self, mock_get):
        fake_quota = mock.Mock(project_id='PROJECT')
        mock_get.return_value = fake_quota
        self._verify('openstack.proxy.Proxy._get', self.proxy.get_quota_default, method_args=['QUOTA_ID'], expected_args=[quota.QuotaDefault], expected_kwargs={'project': fake_quota.id, 'requires_id': False})
        mock_get.assert_called_once_with(quota.Quota, 'QUOTA_ID')

    def test_quotas(self):
        self.verify_list(self.proxy.quotas, quota.Quota)

    def test_quota_update(self):
        self.verify_update(self.proxy.update_quota, quota.Quota)