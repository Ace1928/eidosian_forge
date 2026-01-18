import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, StorageVolume, NodeAuthSSHKey, NodeAuthPassword
from libcloud.test.compute import TestCaseMixin
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.linode import LinodeNodeDriver
class LinodeTest(unittest.TestCase, TestCaseMixin):

    def setUp(self):
        LinodeNodeDriver.connectionCls.conn_class = LinodeMockHttp
        LinodeMockHttp.use_param = 'api_action'
        self.driver = LinodeNodeDriver('foo', api_version='3.0')

    def test_list_nodes(self):
        nodes = self.driver.list_nodes()
        self.assertEqual(len(nodes), 1)
        node = nodes[0]
        self.assertEqual(node.id, '8098')
        self.assertEqual(node.name, 'api-node3')
        self.assertEqual(node.extra['PLANID'], '2')
        self.assertTrue('75.127.96.245' in node.public_ips)
        self.assertEqual(node.private_ips, [])

    def test_reboot_node(self):
        node = self.driver.list_nodes()[0]
        self.driver.reboot_node(node)

    def test_destroy_node(self):
        node = self.driver.list_nodes()[0]
        self.driver.destroy_node(node)

    def test_create_node_password_auth(self):
        self.driver.create_node(name='Test', location=self.driver.list_locations()[0], size=self.driver.list_sizes()[0], image=self.driver.list_images()[6], auth=NodeAuthPassword('test123'))

    def test_create_node_ssh_key_auth(self):
        node = self.driver.create_node(name='Test', location=self.driver.list_locations()[0], size=self.driver.list_sizes()[0], image=self.driver.list_images()[6], auth=NodeAuthSSHKey('foo'))
        self.assertTrue(isinstance(node, Node))

    def test_list_sizes(self):
        sizes = self.driver.list_sizes()
        self.assertEqual(len(sizes), 9)
        for size in sizes:
            self.assertEqual(size.ram, int(size.name.split(' ')[1]))

    def test_list_images(self):
        images = self.driver.list_images()
        self.assertEqual(len(images), 30)

    def test_create_node_response(self):
        node = self.driver.create_node(name='node-name', location=self.driver.list_locations()[0], size=self.driver.list_sizes()[0], image=self.driver.list_images()[0], auth=NodeAuthPassword('foobar'))
        self.assertTrue(isinstance(node, Node))

    def test_destroy_volume(self):
        node = self.driver.list_nodes()[0]
        volume = StorageVolume(id=55648, name='test', size=1024, driver=self.driver, extra={'LINODEID': node.id})
        self.driver.destroy_volume(volume)

    def test_ex_create_volume(self):
        node = self.driver.list_nodes()[0]
        volume = self.driver.ex_create_volume(size=4096, name='Another test image', node=node, fs_type='ext4')
        self.assertTrue(isinstance(volume, StorageVolume))

    def test_ex_list_volumes(self):
        node = self.driver.list_nodes()[0]
        volumes = self.driver.ex_list_volumes(node=node)
        self.assertTrue(isinstance(volumes, list))
        self.assertTrue(isinstance(volumes[0], StorageVolume))
        self.assertEqual(len(volumes), 2)