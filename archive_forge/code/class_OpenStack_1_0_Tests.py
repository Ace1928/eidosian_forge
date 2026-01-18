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
class OpenStack_1_0_Tests(TestCaseMixin, unittest.TestCase):
    should_list_locations = False
    should_list_volumes = False
    driver_klass = OpenStack_1_0_NodeDriver
    driver_args = OPENSTACK_PARAMS
    driver_kwargs = {}

    @classmethod
    def create_driver(self):
        if self is not OpenStack_1_0_FactoryMethodTests:
            self.driver_type = self.driver_klass
        return self.driver_type(*self.driver_args, **self.driver_kwargs)

    def setUp(self):

        def get_endpoint(*args, **kwargs):
            return 'https://servers.api.rackspacecloud.com/v1.0/slug'
        self.driver_klass.connectionCls.get_endpoint = get_endpoint
        self.driver_klass.connectionCls.conn_class = OpenStackMockHttp
        self.driver_klass.connectionCls.auth_url = 'https://auth.api.example.com'
        OpenStackMockHttp.type = None
        self.driver = self.create_driver()
        self.driver.connection._populate_hosts_and_request_paths()
        clear_pricing_data()

    @patch('libcloud.common.openstack.OpenStackServiceCatalog')
    def test_populate_hosts_and_requests_path(self, _):
        tomorrow = datetime.datetime.today() + datetime.timedelta(1)
        cls = self.driver_klass.connectionCls
        count = 5
        con = cls('username', 'key')
        osa = con.get_auth_class()
        mocked_auth_method = Mock()
        osa.authenticate = mocked_auth_method
        for i in range(0, count):
            con._populate_hosts_and_request_paths()
            if i == 0:
                osa.auth_token = '1234'
                osa.auth_token_expires = tomorrow
        self.assertEqual(mocked_auth_method.call_count, 1)
        osa.auth_token = None
        osa.auth_token_expires = None
        con = cls('username', 'key', ex_force_base_url='http://ponies', ex_force_auth_token='1234')
        osa = con.get_auth_class()
        mocked_auth_method = Mock()
        osa.authenticate = mocked_auth_method
        for i in range(0, count):
            con._populate_hosts_and_request_paths()
        self.assertEqual(mocked_auth_method.call_count, 0)

    def test_auth_token_is_set(self):
        self.driver.connection._populate_hosts_and_request_paths()
        self.assertEqual(self.driver.connection.auth_token, 'aaaaaaaaaaaa-bbb-cccccccccccccc')

    def test_auth_token_expires_is_set(self):
        self.driver.connection._populate_hosts_and_request_paths()
        expires = self.driver.connection.auth_token_expires
        self.assertEqual(expires.isoformat(), '2999-11-23T21:00:14-06:00')

    def test_auth(self):
        if self.driver.connection._auth_version == '2.0':
            return
        OpenStackMockHttp.type = 'UNAUTHORIZED'
        try:
            self.driver = self.create_driver()
            self.driver.list_nodes()
        except InvalidCredsError as e:
            self.assertEqual(True, isinstance(e, InvalidCredsError))
        else:
            self.fail('test should have thrown')

    def test_auth_missing_key(self):
        if self.driver.connection._auth_version == '2.0':
            return
        OpenStackMockHttp.type = 'UNAUTHORIZED_MISSING_KEY'
        try:
            self.driver = self.create_driver()
            self.driver.list_nodes()
        except MalformedResponseError as e:
            self.assertEqual(True, isinstance(e, MalformedResponseError))
        else:
            self.fail('test should have thrown')

    def test_auth_server_error(self):
        if self.driver.connection._auth_version == '2.0':
            return
        OpenStackMockHttp.type = 'INTERNAL_SERVER_ERROR'
        try:
            self.driver = self.create_driver()
            self.driver.list_nodes()
        except MalformedResponseError as e:
            self.assertEqual(True, isinstance(e, MalformedResponseError))
        else:
            self.fail('test should have thrown')

    def test_ex_auth_cache_passed_to_identity_connection(self):
        kwargs = self.driver_kwargs.copy()
        kwargs['ex_auth_cache'] = OpenStackMockAuthCache()
        driver = self.driver_type(*self.driver_args, **kwargs)
        driver.list_nodes()
        self.assertEqual(kwargs['ex_auth_cache'], driver.connection.get_auth_class().auth_cache)

    def test_unauthorized_clears_cached_auth_context(self):
        auth_cache = OpenStackMockAuthCache()
        self.assertEqual(len(auth_cache), 0)
        kwargs = self.driver_kwargs.copy()
        kwargs['ex_auth_cache'] = auth_cache
        driver = self.driver_type(*self.driver_args, **kwargs)
        driver.list_nodes()
        self.assertEqual(len(auth_cache), 1)
        self.driver_klass.connectionCls.conn_class.type = 'UNAUTHORIZED'
        with pytest.raises(BaseHTTPError):
            driver.list_nodes()
        self.assertEqual(len(auth_cache), 0)

    def test_error_parsing_when_body_is_missing_message(self):
        OpenStackMockHttp.type = 'NO_MESSAGE_IN_ERROR_BODY'
        try:
            self.driver.list_images()
        except Exception as e:
            self.assertEqual(True, isinstance(e, Exception))
        else:
            self.fail('test should have thrown')

    def test_list_locations(self):
        locations = self.driver.list_locations()
        self.assertEqual(len(locations), 1)

    def test_list_nodes(self):
        OpenStackMockHttp.type = 'EMPTY'
        ret = self.driver.list_nodes()
        self.assertEqual(len(ret), 0)
        OpenStackMockHttp.type = None
        ret = self.driver.list_nodes()
        self.assertEqual(len(ret), 1)
        node = ret[0]
        self.assertEqual('67.23.21.33', node.public_ips[0])
        self.assertTrue('10.176.168.218' in node.private_ips)
        self.assertEqual(node.extra.get('flavorId'), '1')
        self.assertEqual(node.extra.get('imageId'), '11')
        self.assertEqual(type(node.extra.get('metadata')), type(dict()))
        OpenStackMockHttp.type = 'METADATA'
        ret = self.driver.list_nodes()
        self.assertEqual(len(ret), 1)
        node = ret[0]
        self.assertEqual(type(node.extra.get('metadata')), type(dict()))
        self.assertEqual(node.extra.get('metadata').get('somekey'), 'somevalue')
        OpenStackMockHttp.type = None

    def test_list_images(self):
        ret = self.driver.list_images()
        expected = {10: {'serverId': None, 'status': 'ACTIVE', 'created': '2009-07-20T09:14:37-05:00', 'updated': '2009-07-20T09:14:37-05:00', 'progress': None, 'minDisk': None, 'minRam': None}, 11: {'serverId': '91221', 'status': 'ACTIVE', 'created': '2009-11-29T20:22:09-06:00', 'updated': '2009-11-29T20:24:08-06:00', 'progress': '100', 'minDisk': '5', 'minRam': '256'}}
        for ret_idx, extra in list(expected.items()):
            for key, value in list(extra.items()):
                self.assertEqual(ret[ret_idx].extra[key], value)

    def test_create_node(self):
        image = NodeImage(id=11, name='Ubuntu 8.10 (intrepid)', driver=self.driver)
        size = NodeSize(1, '256 slice', None, None, None, None, driver=self.driver)
        node = self.driver.create_node(name='racktest', image=image, size=size)
        self.assertEqual(node.name, 'racktest')
        self.assertEqual(node.extra.get('password'), 'racktestvJq7d3')

    def test_create_node_without_adminPass(self):
        OpenStackMockHttp.type = 'NO_ADMIN_PASS'
        image = NodeImage(id=11, name='Ubuntu 8.10 (intrepid)', driver=self.driver)
        size = NodeSize(1, '256 slice', None, None, None, None, driver=self.driver)
        node = self.driver.create_node(name='racktest', image=image, size=size)
        self.assertEqual(node.name, 'racktest')
        self.assertIsNone(node.extra.get('password'))

    def test_create_node_ex_shared_ip_group(self):
        OpenStackMockHttp.type = 'EX_SHARED_IP_GROUP'
        image = NodeImage(id=11, name='Ubuntu 8.10 (intrepid)', driver=self.driver)
        size = NodeSize(1, '256 slice', None, None, None, None, driver=self.driver)
        node = self.driver.create_node(name='racktest', image=image, size=size, ex_shared_ip_group_id='12345')
        self.assertEqual(node.name, 'racktest')
        self.assertEqual(node.extra.get('password'), 'racktestvJq7d3')

    def test_create_node_with_metadata(self):
        OpenStackMockHttp.type = 'METADATA'
        image = NodeImage(id=11, name='Ubuntu 8.10 (intrepid)', driver=self.driver)
        size = NodeSize(1, '256 slice', None, None, None, None, driver=self.driver)
        metadata = {'a': 'b', 'c': 'd'}
        files = {'/file1': 'content1', '/file2': 'content2'}
        node = self.driver.create_node(name='racktest', image=image, size=size, ex_metadata=metadata, ex_files=files)
        self.assertEqual(node.name, 'racktest')
        self.assertEqual(node.extra.get('password'), 'racktestvJq7d3')
        self.assertEqual(node.extra.get('metadata'), metadata)

    def test_reboot_node(self):
        node = Node(id=72258, name=None, state=None, public_ips=None, private_ips=None, driver=self.driver)
        ret = node.reboot()
        self.assertTrue(ret is True)

    def test_destroy_node(self):
        node = Node(id=72258, name=None, state=None, public_ips=None, private_ips=None, driver=self.driver)
        ret = node.destroy()
        self.assertTrue(ret is True)

    def test_ex_limits(self):
        limits = self.driver.ex_limits()
        self.assertTrue('rate' in limits)
        self.assertTrue('absolute' in limits)

    def test_create_image(self):
        node = Node(id=444222, name=None, state=None, public_ips=None, private_ips=None, driver=self.driver)
        image = self.driver.create_image(node, 'imgtest')
        self.assertEqual(image.name, 'imgtest')
        self.assertEqual(image.id, '12345')

    def test_delete_image(self):
        image = NodeImage(id=333111, name='Ubuntu 8.10 (intrepid)', driver=self.driver)
        ret = self.driver.delete_image(image)
        self.assertTrue(ret)

    def test_ex_list_ip_addresses(self):
        ret = self.driver.ex_list_ip_addresses(node_id=72258)
        self.assertEqual(2, len(ret.public_addresses))
        self.assertTrue('67.23.10.131' in ret.public_addresses)
        self.assertTrue('67.23.10.132' in ret.public_addresses)
        self.assertEqual(1, len(ret.private_addresses))
        self.assertTrue('10.176.42.16' in ret.private_addresses)

    def test_ex_list_ip_groups(self):
        ret = self.driver.ex_list_ip_groups()
        self.assertEqual(2, len(ret))
        self.assertEqual('1234', ret[0].id)
        self.assertEqual('Shared IP Group 1', ret[0].name)
        self.assertEqual('5678', ret[1].id)
        self.assertEqual('Shared IP Group 2', ret[1].name)
        self.assertTrue(ret[0].servers is None)

    def test_ex_list_ip_groups_detail(self):
        ret = self.driver.ex_list_ip_groups(details=True)
        self.assertEqual(2, len(ret))
        self.assertEqual('1234', ret[0].id)
        self.assertEqual('Shared IP Group 1', ret[0].name)
        self.assertEqual(2, len(ret[0].servers))
        self.assertEqual('422', ret[0].servers[0])
        self.assertEqual('3445', ret[0].servers[1])
        self.assertEqual('5678', ret[1].id)
        self.assertEqual('Shared IP Group 2', ret[1].name)
        self.assertEqual(3, len(ret[1].servers))
        self.assertEqual('23203', ret[1].servers[0])
        self.assertEqual('2456', ret[1].servers[1])
        self.assertEqual('9891', ret[1].servers[2])

    def test_ex_create_ip_group(self):
        ret = self.driver.ex_create_ip_group('Shared IP Group 1', '5467')
        self.assertEqual('1234', ret.id)
        self.assertEqual('Shared IP Group 1', ret.name)
        self.assertEqual(1, len(ret.servers))
        self.assertEqual('422', ret.servers[0])

    def test_ex_delete_ip_group(self):
        ret = self.driver.ex_delete_ip_group('5467')
        self.assertEqual(True, ret)

    def test_ex_share_ip(self):
        ret = self.driver.ex_share_ip('1234', '3445', '67.23.21.133')
        self.assertEqual(True, ret)

    def test_ex_unshare_ip(self):
        ret = self.driver.ex_unshare_ip('3445', '67.23.21.133')
        self.assertEqual(True, ret)

    def test_ex_resize(self):
        node = Node(id=444222, name=None, state=None, public_ips=None, private_ips=None, driver=self.driver)
        size = NodeSize(1, '256 slice', None, None, None, None, driver=self.driver)
        self.assertTrue(self.driver.ex_resize(node=node, size=size))

    def test_ex_confirm_resize(self):
        node = Node(id=444222, name=None, state=None, public_ips=None, private_ips=None, driver=self.driver)
        self.assertTrue(self.driver.ex_confirm_resize(node=node))

    def test_ex_revert_resize(self):
        node = Node(id=444222, name=None, state=None, public_ips=None, private_ips=None, driver=self.driver)
        self.assertTrue(self.driver.ex_revert_resize(node=node))

    def test_list_sizes(self):
        sizes = self.driver.list_sizes()
        self.assertEqual(len(sizes), 7, 'Wrong sizes count')
        for size in sizes:
            self.assertTrue(isinstance(size.price, float), 'Wrong size price type')
            if self.driver.api_name == 'openstack':
                self.assertEqual(size.price, 0, 'Size price should be zero by default')

    def test_list_sizes_with_specified_pricing(self):
        if self.driver.api_name != 'openstack':
            return
        pricing = {str(i): i for i in range(1, 8)}
        set_pricing(driver_type='compute', driver_name='openstack', pricing=pricing)
        sizes = self.driver.list_sizes()
        self.assertEqual(len(sizes), 7, 'Wrong sizes count')
        for size in sizes:
            self.assertTrue(isinstance(size.price, float), 'Wrong size price type')
            self.assertEqual(float(size.price), float(pricing[size.id]))