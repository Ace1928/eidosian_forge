import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import VPSNET_PARAMS
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vpsnet import VPSNetNodeDriver
class VPSNetTests(unittest.TestCase, TestCaseMixin):

    def setUp(self):
        VPSNetNodeDriver.connectionCls.conn_class = VPSNetMockHttp
        self.driver = VPSNetNodeDriver(*VPSNET_PARAMS)

    def test_create_node(self):
        VPSNetMockHttp.type = 'create'
        image = self.driver.list_images()[0]
        size = self.driver.list_sizes()[0]
        node = self.driver.create_node('foo', image, size)
        self.assertEqual(node.name, 'foo')

    def test_list_nodes(self):
        VPSNetMockHttp.type = 'virtual_machines'
        node = self.driver.list_nodes()[0]
        self.assertEqual(node.id, '1384')
        self.assertEqual(node.state, NodeState.RUNNING)

    def test_reboot_node(self):
        VPSNetMockHttp.type = 'virtual_machines'
        node = self.driver.list_nodes()[0]
        VPSNetMockHttp.type = 'reboot'
        ret = self.driver.reboot_node(node)
        self.assertEqual(ret, True)

    def test_destroy_node(self):
        VPSNetMockHttp.type = 'delete'
        node = Node('2222', None, None, None, None, self.driver)
        ret = self.driver.destroy_node(node)
        self.assertTrue(ret)
        VPSNetMockHttp.type = 'delete_fail'
        node = Node('2223', None, None, None, None, self.driver)
        self.assertRaises(Exception, self.driver.destroy_node, node)

    def test_list_images(self):
        VPSNetMockHttp.type = 'templates'
        ret = self.driver.list_images()
        self.assertEqual(ret[0].id, '9')
        self.assertEqual(ret[-1].id, '160')

    def test_list_sizes(self):
        VPSNetMockHttp.type = 'sizes'
        ret = self.driver.list_sizes()
        self.assertEqual(len(ret), 1)
        self.assertEqual(ret[0].id, '1')
        self.assertEqual(ret[0].name, '1 Node')

    def test_destroy_node_response(self):
        node = Node('2222', None, None, None, None, self.driver)
        VPSNetMockHttp.type = 'delete'
        ret = self.driver.destroy_node(node)
        self.assertTrue(isinstance(ret, bool))

    def test_reboot_node_response(self):
        VPSNetMockHttp.type = 'virtual_machines'
        node = self.driver.list_nodes()[0]
        VPSNetMockHttp.type = 'reboot'
        ret = self.driver.reboot_node(node)
        self.assertTrue(isinstance(ret, bool))