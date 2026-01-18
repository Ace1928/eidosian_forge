import os
import sys
import datetime
import unittest
from unittest import mock
from unittest.mock import Mock, patch
import pytest
import requests_mock
from libcloud.test import XML_HEADERS, MockHttp
from libcloud.pricing import set_pricing, clear_pricing_data
from libcloud.utils.py3 import u, httplib, method_type
from libcloud.common.base import LibcloudConnection
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.compute.base import Node, NodeSize, NodeImage
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import OPENSTACK_PARAMS
from libcloud.compute.types import (
from libcloud.utils.iso8601 import UTC
from libcloud.common.exceptions import BaseHTTPError
from libcloud.compute.providers import get_driver
from libcloud.test.file_fixtures import OpenStackFixtures, ComputeFileFixtures
from libcloud.common.openstack_identity import (
from libcloud.compute.drivers.openstack import (
class OpenStack_2_Tests(OpenStack_1_1_Tests):
    driver_klass = OpenStack_2_NodeDriver
    driver_type = OpenStack_2_NodeDriver
    driver_kwargs = {'ex_force_auth_version': '2.0', 'ex_force_auth_url': 'https://auth.api.example.com'}

    def setUp(self):
        super().setUp()
        self.driver_klass.image_connectionCls.conn_class = OpenStack_2_0_MockHttp
        self.driver_klass.image_connectionCls.auth_url = 'https://auth.api.example.com'
        self.driver.image_connection._populate_hosts_and_request_paths()
        self.driver_klass.network_connectionCls.conn_class = OpenStack_2_0_MockHttp
        self.driver_klass.network_connectionCls.auth_url = 'https://auth.api.example.com'
        self.driver.network_connection._populate_hosts_and_request_paths()
        self.driver_klass.volumev2_connectionCls.conn_class = OpenStack_2_0_MockHttp
        self.driver_klass.volumev2_connectionCls.auth_url = 'https://auth.api.example.com'
        self.driver.volumev2_connection._populate_hosts_and_request_paths()
        self.driver_klass.volumev3_connectionCls.conn_class = OpenStack_2_0_MockHttp
        self.driver_klass.volumev3_connectionCls.auth_url = 'https://auth.api.example.com'
        self.driver.volumev3_connection._populate_hosts_and_request_paths()

    def test__paginated_request_single_page(self):
        snapshots = self.driver._paginated_request('/snapshots/detail', 'snapshots', self.driver._get_volume_connection())['snapshots']
        self.assertEqual(len(snapshots), 3)
        self.assertEqual(snapshots[0]['name'], 'snap-001')

    def test__paginated_request_two_pages(self):
        snapshots = self.driver._paginated_request('/snapshots/detail?unit_test=paginate', 'snapshots', self.driver._get_volume_connection())['snapshots']
        self.assertEqual(len(snapshots), 6)
        self.assertEqual(snapshots[0]['name'], 'snap-101')
        self.assertEqual(snapshots[3]['name'], 'snap-001')

    def test_list_images_with_pagination_invalid_response_no_infinite_loop(self):
        OpenStack_2_0_MockHttp.type = 'invalid_next'
        ret = self.driver.list_images()
        self.assertEqual(len(ret), 2)

    @mock.patch('libcloud.compute.drivers.openstack.PAGINATION_LIMIT', 10)
    def test__paginated_request_raises_if_stuck_in_a_loop(self):
        with pytest.raises(OpenStackException):
            self.driver._paginated_request('/snapshots/detail?unit_test=pagination_loop', 'snapshots', self.driver._get_volume_connection())

    def test_ex_force_auth_token_passed_to_connection(self):
        base_url = 'https://servers.api.rackspacecloud.com/v1.1/slug'
        kwargs = {'ex_force_auth_version': '2.0', 'ex_force_auth_token': 'preset-auth-token', 'ex_force_auth_url': 'https://auth.api.example.com', 'ex_force_base_url': base_url}
        driver = self.driver_type(*self.driver_args, **kwargs)
        driver.list_nodes()
        self.assertEqual(kwargs['ex_force_auth_token'], driver.connection.auth_token)
        self.assertEqual('servers.api.rackspacecloud.com', driver.connection.host)
        self.assertEqual('/v1.1/slug', driver.connection.request_path)
        self.assertEqual(443, driver.connection.port)

    def test_get_image(self):
        image_id = 'f24a3c1b-d52a-4116-91da-25b3eee8f55e'
        image = self.driver.get_image(image_id)
        self.assertEqual(image.id, image_id)
        self.assertEqual(image.name, 'hypernode')
        self.assertIsNone(image.extra['serverId'])
        self.assertEqual(image.extra['minDisk'], 40)
        self.assertEqual(image.extra['minRam'], 0)
        self.assertEqual(image.extra['visibility'], 'shared')

    def test_list_images(self):
        images = self.driver.list_images()
        self.assertEqual(len(images), 3, 'Wrong images count')
        image = images[0]
        self.assertEqual(image.id, 'f24a3c1b-d52a-4116-91da-25b3eee8f55e')
        self.assertEqual(image.name, 'hypernode')
        self.assertEqual(image.extra['updated'], '2017-11-28T10:19:49Z')
        self.assertEqual(image.extra['created'], '2017-09-11T13:00:05Z')
        self.assertEqual(image.extra['status'], 'active')
        self.assertEqual(image.extra['os_type'], 'linux')
        self.assertIsNone(image.extra['serverId'])
        self.assertEqual(image.extra['minDisk'], 40)
        self.assertEqual(image.extra['minRam'], 0)

    def test_ex_update_image(self):
        image_id = 'f24a3c1b-d52a-4116-91da-25b3eee8f55e'
        data = {'op': 'replace', 'path': '/visibility', 'value': 'shared'}
        image = self.driver.ex_update_image(image_id, data)
        self.assertEqual(image.name, 'hypernode')
        self.assertIsNone(image.extra['serverId'])
        self.assertEqual(image.extra['minDisk'], 40)
        self.assertEqual(image.extra['minRam'], 0)
        self.assertEqual(image.extra['visibility'], 'shared')

    def test_ex_list_image_members(self):
        image_id = 'd9a9cd9a-278a-444c-90a6-d24b8c688a63'
        image_member_id = '016926dff12345e8b10329f24c99745b'
        image_members = self.driver.ex_list_image_members(image_id)
        self.assertEqual(len(image_members), 30, 'Wrong image member count')
        image_member = image_members[0]
        self.assertEqual(image_member.id, image_member_id)
        self.assertEqual(image_member.image_id, image_id)
        self.assertEqual(image_member.state, NodeImageMemberState.ACCEPTED)
        self.assertEqual(image_member.created, '2017-01-12T12:31:50Z')
        self.assertEqual(image_member.extra['updated'], '2017-01-12T12:31:54Z')
        self.assertEqual(image_member.extra['schema'], '/v2/schemas/member')

    def test_ex_create_image_member(self):
        image_id = '9af1a54e-a1b2-4df8-b747-4bec97abc799'
        image_member_id = 'e2151b1fe02d4a8a2d1f5fc331522c0a'
        image_member = self.driver.ex_create_image_member(image_id, image_member_id)
        self.assertEqual(image_member.id, image_member_id)
        self.assertEqual(image_member.image_id, image_id)
        self.assertEqual(image_member.state, NodeImageMemberState.PENDING)
        self.assertEqual(image_member.created, '2018-03-02T14:19:38Z')
        self.assertEqual(image_member.extra['updated'], '2018-03-02T14:19:38Z')
        self.assertEqual(image_member.extra['schema'], '/v2/schemas/member')

    def test_ex_get_image_member(self):
        image_id = 'd9a9cd9a-278a-444c-90a6-d24b8c688a63'
        image_member_id = '016926dff12345e8b10329f24c99745b'
        image_member = self.driver.ex_get_image_member(image_id, image_member_id)
        self.assertEqual(image_member.id, image_member_id)
        self.assertEqual(image_member.image_id, image_id)
        self.assertEqual(image_member.state, NodeImageMemberState.ACCEPTED)
        self.assertEqual(image_member.created, '2017-01-12T12:31:50Z')
        self.assertEqual(image_member.extra['updated'], '2017-01-12T12:31:54Z')
        self.assertEqual(image_member.extra['schema'], '/v2/schemas/member')

    def test_ex_accept_image_member(self):
        image_id = '8af1a54e-a1b2-4df8-b747-4bec97abc799'
        image_member_id = 'e2151b1fe02d4a8a2d1f5fc331522c0a'
        image_member = self.driver.ex_accept_image_member(image_id, image_member_id)
        self.assertEqual(image_member.id, image_member_id)
        self.assertEqual(image_member.image_id, image_id)
        self.assertEqual(image_member.state, NodeImageMemberState.ACCEPTED)
        self.assertEqual(image_member.created, '2018-03-02T14:19:38Z')
        self.assertEqual(image_member.extra['updated'], '2018-03-02T14:20:37Z')
        self.assertEqual(image_member.extra['schema'], '/v2/schemas/member')

    def test_ex_list_networks(self):
        networks = self.driver.ex_list_networks()
        network = networks[0]
        self.assertEqual(len(networks), 2)
        self.assertEqual(network.name, 'net1')
        self.assertEqual(network.extra['subnets'], ['54d6f61d-db07-451c-9ab3-b9609b6b6f0b'])

    def test_ex_get_network(self):
        network = self.driver.ex_get_network('cc2dad14-827a-feea-416b-f13e50511a0a')
        self.assertEqual(network.id, 'cc2dad14-827a-feea-416b-f13e50511a0a')
        self.assertTrue(isinstance(network, OpenStackNetwork))
        self.assertEqual(network.name, 'net2')

    def test_ex_list_subnets(self):
        subnets = self.driver.ex_list_subnets()
        subnet = subnets[0]
        self.assertEqual(len(subnets), 2)
        self.assertEqual(subnet.name, 'private-subnet')
        self.assertEqual(subnet.cidr, '10.0.0.0/24')

    def test_ex_create_subnet(self):
        network = self.driver.ex_list_networks()[0]
        subnet = self.driver.ex_create_subnet('name', network, '10.0.0.0/24', ip_version=4, dns_nameservers=['10.0.0.01'])
        self.assertEqual(subnet.name, 'name')
        self.assertEqual(subnet.cidr, '10.0.0.0/24')

    def test_ex_delete_subnet(self):
        subnet = self.driver.ex_list_subnets()[0]
        self.assertTrue(self.driver.ex_delete_subnet(subnet=subnet))

    def test_ex_update_subnet(self):
        subnet = self.driver.ex_list_subnets()[0]
        subnet = self.driver.ex_update_subnet(subnet, name='net2')
        self.assertEqual(subnet.name, 'name')

    def test_ex_list_network(self):
        networks = self.driver.ex_list_networks()
        network = networks[0]
        self.assertEqual(len(networks), 2)
        self.assertEqual(network.name, 'net1')

    def test_ex_create_network(self):
        network = self.driver.ex_create_network(name='net1', cidr='127.0.0.0/24')
        self.assertEqual(network.name, 'net1')

    def test_ex_delete_network(self):
        network = self.driver.ex_list_networks()[0]
        self.assertTrue(self.driver.ex_delete_network(network=network))

    def test_ex_list_ports(self):
        ports = self.driver.ex_list_ports()
        port = ports[0]
        self.assertEqual(port.id, '126da55e-cfcb-41c8-ae39-a26cb8a7e723')
        self.assertEqual(port.state, OpenStack_2_PortInterfaceState.BUILD)
        self.assertEqual(port.created, '2018-07-04T14:38:18Z')
        self.assertEqual(port.extra['network_id'], '123c8a8c-6427-4e8f-a805-2035365f4d43')
        self.assertEqual(port.extra['project_id'], 'abcdec85bee34bb0a44ab8255eb36abc')
        self.assertEqual(port.extra['tenant_id'], 'abcdec85bee34bb0a44ab8255eb36abc')
        self.assertEqual(port.extra['name'], '')

    def test_ex_create_port(self):
        network = OpenStackNetwork(id='123c8a8c-6427-4e8f-a805-2035365f4d43', name='test-network', cidr='1.2.3.4', driver=self.driver)
        port = self.driver.ex_create_port(network=network, description='Some port description', name='Some port name', admin_state_up=True)
        self.assertEqual(port.id, '126da55e-cfcb-41c8-ae39-a26cb8a7e723')
        self.assertEqual(port.state, OpenStack_2_PortInterfaceState.BUILD)
        self.assertEqual(port.created, '2018-07-04T14:38:18Z')
        self.assertEqual(port.extra['network_id'], '123c8a8c-6427-4e8f-a805-2035365f4d43')
        self.assertEqual(port.extra['project_id'], 'abcdec85bee34bb0a44ab8255eb36abc')
        self.assertEqual(port.extra['tenant_id'], 'abcdec85bee34bb0a44ab8255eb36abc')
        self.assertEqual(port.extra['admin_state_up'], True)
        self.assertEqual(port.extra['name'], 'Some port name')
        self.assertEqual(port.extra['description'], 'Some port description')

    def test_ex_get_port(self):
        port = self.driver.ex_get_port('126da55e-cfcb-41c8-ae39-a26cb8a7e723')
        self.assertEqual(port.id, '126da55e-cfcb-41c8-ae39-a26cb8a7e723')
        self.assertEqual(port.state, OpenStack_2_PortInterfaceState.BUILD)
        self.assertEqual(port.created, '2018-07-04T14:38:18Z')
        self.assertEqual(port.extra['network_id'], '123c8a8c-6427-4e8f-a805-2035365f4d43')
        self.assertEqual(port.extra['project_id'], 'abcdec85bee34bb0a44ab8255eb36abc')
        self.assertEqual(port.extra['tenant_id'], 'abcdec85bee34bb0a44ab8255eb36abc')
        self.assertEqual(port.extra['name'], 'Some port name')

    def test_ex_delete_port(self):
        ports = self.driver.ex_list_ports()
        port = ports[0]
        ret = self.driver.ex_delete_port(port)
        self.assertTrue(ret)

    def test_ex_update_port(self):
        port = self.driver.ex_get_port('126da55e-cfcb-41c8-ae39-a26cb8a7e723')
        ret = self.driver.ex_update_port(port, port_security_enabled=False)
        self.assertEqual(ret.extra['name'], 'Some port name')

    def test_ex_update_port_allowed_address_pairs(self):
        allowed_address_pairs = [{'ip_address': '1.2.3.4'}, {'ip_address': '2.3.4.5'}]
        port = self.driver.ex_get_port('126da55e-cfcb-41c8-ae39-a26cb8a7e723')
        ret = self.driver.ex_update_port(port, allowed_address_pairs=allowed_address_pairs)
        self.assertEqual(ret.extra['allowed_address_pairs'], allowed_address_pairs)

    def test_detach_port_interface(self):
        node = Node(id='1c01300f-ef97-4937-8f03-ac676d6234be', name=None, state=None, public_ips=None, private_ips=None, driver=self.driver)
        ports = self.driver.ex_list_ports()
        port = ports[0]
        ret = self.driver.ex_detach_port_interface(node, port)
        self.assertTrue(ret)

    def test_attach_port_interface(self):
        node = Node(id='1c01300f-ef97-4937-8f03-ac676d6234be', name=None, state=None, public_ips=None, private_ips=None, driver=self.driver)
        ports = self.driver.ex_list_ports()
        port = ports[0]
        ret = self.driver.ex_attach_port_interface(node, port)
        self.assertTrue(ret)

    def test_list_volumes(self):
        volumes = self.driver.list_volumes()
        self.assertEqual(len(volumes), 2)
        volume = volumes[0]
        self.assertEqual('6edbc2f4-1507-44f8-ac0d-eed1d2608d38', volume.id)
        self.assertEqual('test-volume-attachments', volume.name)
        self.assertEqual(StorageVolumeState.INUSE, volume.state)
        self.assertEqual(2, volume.size)
        self.assertEqual(volume.extra, {'description': '', 'attachments': [{'attachment_id': '3b4db356-253d-4fab-bfa0-e3626c0b8405', 'id': '6edbc2f4-1507-44f8-ac0d-eed1d2608d38', 'device': '/dev/vdb', 'server_id': 'f4fda93b-06e0-4743-8117-bc8bcecd651b', 'volume_id': '6edbc2f4-1507-44f8-ac0d-eed1d2608d38'}], 'snapshot_id': None, 'state': 'in-use', 'location': 'nova', 'volume_type': 'lvmdriver-1', 'metadata': {}, 'created_at': '2013-06-24T11:20:13.000000'})
        volume = volumes[1]
        self.assertEqual('cfcec3bc-b736-4db5-9535-4c24112691b5', volume.id)
        self.assertEqual('test_volume', volume.name)
        self.assertEqual(50, volume.size)
        self.assertEqual(StorageVolumeState.UNKNOWN, volume.state)
        self.assertEqual(volume.extra, {'description': 'some description', 'attachments': [], 'snapshot_id': '01f48111-7866-4cd2-986a-e92683c4a363', 'state': 'some-unknown-state', 'location': 'nova', 'volume_type': None, 'metadata': {}, 'created_at': '2013-06-21T12:39:02.000000'})

    def test_create_volume_passes_location_to_request_only_if_not_none(self):
        with patch.object(self.driver._get_volume_connection(), 'request') as mock_request:
            self.driver.create_volume(1, 'test', location='mylocation')
            name, args, kwargs = mock_request.mock_calls[0]
            self.assertEqual(kwargs['data']['volume']['availability_zone'], 'mylocation')

    def test_create_volume_does_not_pass_location_to_request_if_none(self):
        with patch.object(self.driver._get_volume_connection(), 'request') as mock_request:
            self.driver.create_volume(1, 'test')
            name, args, kwargs = mock_request.mock_calls[0]
            self.assertFalse('availability_zone' in kwargs['data']['volume'])

    def test_create_volume_passes_volume_type_to_request_only_if_not_none(self):
        with patch.object(self.driver._get_volume_connection(), 'request') as mock_request:
            self.driver.create_volume(1, 'test', ex_volume_type='myvolumetype')
            name, args, kwargs = mock_request.mock_calls[0]
            self.assertEqual(kwargs['data']['volume']['volume_type'], 'myvolumetype')

    def test_create_volume_does_not_pass_volume_type_to_request_if_none(self):
        with patch.object(self.driver._get_volume_connection(), 'request') as mock_request:
            self.driver.create_volume(1, 'test')
            name, args, kwargs = mock_request.mock_calls[0]
            self.assertFalse('volume_type' in kwargs['data']['volume'])

    def test_create_volume_passes_image_ref_to_request_only_if_not_none(self):
        with patch.object(self.driver._get_volume_connection(), 'request') as mock_request:
            self.driver.create_volume(1, 'test', ex_image_ref='353c4bd2-b28f-4857-9b7b-808db4397d03')
            name, args, kwargs = mock_request.mock_calls[0]
            self.assertEqual(kwargs['data']['volume']['imageRef'], '353c4bd2-b28f-4857-9b7b-808db4397d03')

    def test_create_volume_does_not_pass_image_ref_to_request_if_none(self):
        with patch.object(self.driver._get_volume_connection(), 'request') as mock_request:
            self.driver.create_volume(1, 'test')
            name, args, kwargs = mock_request.mock_calls[0]
            self.assertFalse('imageRef' in kwargs['data']['volume'])

    def test_ex_create_snapshot_does_not_post_optional_parameters_if_none(self):
        volume = self.driver.list_volumes()[0]
        with patch.object(self.driver, '_to_snapshot'):
            with patch.object(self.driver._get_volume_connection(), 'request') as mock_request:
                self.driver.create_volume_snapshot(volume, name=None, ex_description=None, ex_force=True)
        name, args, kwargs = mock_request.mock_calls[0]
        self.assertFalse('display_name' in kwargs['data']['snapshot'])
        self.assertFalse('display_description' in kwargs['data']['snapshot'])

    def test_ex_list_routers(self):
        routers = self.driver.ex_list_routers()
        router = routers[0]
        self.assertEqual(len(routers), 2)
        self.assertEqual(router.name, 'router2')
        self.assertEqual(router.status, 'ACTIVE')
        self.assertEqual(router.extra['routes'], [{'destination': '179.24.1.0/24', 'nexthop': '172.24.3.99'}])

    def test_ex_create_router(self):
        router = self.driver.ex_create_router('router1', admin_state_up=True)
        self.assertEqual(router.name, 'router1')

    def test_ex_delete_router(self):
        router = self.driver.ex_list_routers()[1]
        self.assertTrue(self.driver.ex_delete_router(router=router))

    def test_manage_router_interfaces(self):
        router = self.driver.ex_list_routers()[1]
        port = self.driver.ex_list_ports()[0]
        subnet = self.driver.ex_list_subnets()[0]
        self.assertTrue(self.driver.ex_add_router_port(router, port))
        self.assertTrue(self.driver.ex_del_router_port(router, port))
        self.assertTrue(self.driver.ex_add_router_subnet(router, subnet))
        self.assertTrue(self.driver.ex_del_router_subnet(router, subnet))

    def test_detach_volume(self):
        node = self.driver.list_nodes()[0]
        volume = self.driver.ex_get_volume('abc6a3a1-c4ce-40f6-9b9f-07a61508938d')
        self.assertEqual(self.driver.attach_volume(node, volume, '/dev/sdb'), True)
        self.assertEqual(self.driver.detach_volume(volume), True)

    def test_ex_remove_security_group_from_node(self):
        security_group = OpenStackSecurityGroup('sgid', None, 'sgname', '', self.driver)
        node = Node('1000', 'node', None, [], [], self.driver)
        ret = self.driver.ex_remove_security_group_from_node(security_group, node)
        self.assertTrue(ret)

    def test_force_net_url(self):
        d = OpenStack_2_NodeDriver('user', 'correct_password', ex_force_auth_version='2.0_password', ex_force_auth_url='http://x.y.z.y:5000', ex_force_network_url='http://network.com:9696', ex_tenant_name='admin')
        self.assertEqual(d._ex_force_base_url, None)

    def test_ex_get_quota_set(self):
        quota_set = self.driver.ex_get_quota_set('tenant_id')
        self.assertEqual(quota_set.cores.limit, 20)
        self.assertEqual(quota_set.cores.in_use, 1)
        self.assertEqual(quota_set.cores.reserved, 0)

    def test_ex_get_network_quota(self):
        quota_set = self.driver.ex_get_network_quotas('tenant_id')
        self.assertEqual(quota_set.floatingip.limit, 2)
        self.assertEqual(quota_set.floatingip.in_use, 1)
        self.assertEqual(quota_set.floatingip.reserved, 0)

    def test_ex_get_volume_quota(self):
        quota_set = self.driver.ex_get_volume_quotas('tenant_id')
        self.assertEqual(quota_set.gigabytes.limit, 1000)
        self.assertEqual(quota_set.gigabytes.in_use, 10)
        self.assertEqual(quota_set.gigabytes.reserved, 0)

    def test_ex_list_server_groups(self):
        server_groups = self.driver.ex_list_server_groups()
        self.assertEqual(len(server_groups), 2)
        self.assertEqual(server_groups[1].name, 'server_group_name')

    def test_ex_get_server_group(self):
        server_group = self.driver.ex_get_server_group('616fb98f-46ca-475e-917e-2563e5a8cd19')
        self.assertEqual(server_group.name, 'server_group_name')
        self.assertEqual(server_group.policy, 'anti-affinity')

    def test_ex_del_server_group(self):
        server_group = OpenStack_2_ServerGroup('616fb98f-46ca-475e-917e-2563e5a8cd19', 'name', 'anti-affinity')
        res = self.driver.ex_del_server_group(server_group)
        self.assertTrue(res)

    def test_ex_add_server_group(self):
        server_group = self.driver.ex_add_server_group('server_group_name', 'anti-affinity')
        self.assertEqual(server_group.name, 'server_group_name')
        self.assertEqual(server_group.policy, 'anti-affinity')

    def test_ex_list_floating_ips(self):
        ret = self.driver.ex_list_floating_ips()
        self.assertEqual(ret[0].id, '09ea1784-2f81-46dc-8c91-244b4df75bde')
        self.assertEqual(ret[0].get_pool(), None)
        self.assertEqual(ret[0].ip_address, '10.3.1.42')
        self.assertEqual(ret[0].get_node_id(), None)
        self.assertEqual(ret[1].id, '04c5336a-0629-4694-ba30-04b0bdfa88a4')
        self.assertEqual(ret[1].get_pool(), None)
        self.assertEqual(ret[1].ip_address, '10.3.1.1')
        self.assertEqual(ret[1].get_node_id(), 'fcfc96da-19e2-40fd-8497-f29da1b21143')
        self.assertEqual(ret[2].id, '123c5336a-0629-4694-ba30-04b0bdfa88a4')
        self.assertEqual(ret[2].get_pool(), None)
        self.assertEqual(ret[2].ip_address, '10.3.1.2')
        self.assertEqual(ret[2].get_node_id(), 'cb4fba64-19e2-40fd-8497-f29da1b21143')
        self.assertEqual(ret[3].id, '123c5336a-0629-4694-ba30-04b0bdfa88a4')
        self.assertEqual(ret[3].get_pool(), None)
        self.assertEqual(ret[3].ip_address, '10.3.1.3')
        self.assertEqual(ret[3].get_node_id(), 'cb4fba64-19e2-40fd-8497-f29da1b21143')
        self.assertEqual(ret[4].id, '123c5336a-0629-4694-ba30-04b0bdfa88a4')
        self.assertEqual(ret[4].get_pool(), None)
        self.assertEqual(ret[4].ip_address, '10.3.1.5')
        self.assertEqual(ret[4].get_node_id(), 'cb4fba64-19e2-40fd-8497-f29da1b21143')

    def test_ex_get_floating_ip(self):
        float_ip = self.driver.ex_get_floating_ip('10.0.0.1')
        self.assertEqual(float_ip.ip_address, '10.3.1.21')
        self.assertEqual(float_ip.id, '04c5336a-0629-4694-ba30-04b0bdfa88a4')

    def test_ex_create_floating_ip(self):
        ret = self.driver.ex_create_floating_ip('public')
        self.assertEqual(ret.id, '09ea1784-2f81-46dc-8c91-244b4df75bde')
        self.assertEqual(ret.pool.name, 'public')
        self.assertEqual(ret.ip_address, '10.3.1.42')
        self.assertEqual(ret.node_id, None)

    def test_ex_delete_floating_ip(self):
        ip = OpenStack_1_1_FloatingIpAddress('foo-bar-id', '42.42.42.42', None)
        self.assertTrue(self.driver.ex_delete_floating_ip(ip))