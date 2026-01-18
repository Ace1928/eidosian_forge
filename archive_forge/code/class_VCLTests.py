import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib, xmlrpclib
from libcloud.test.secrets import VCL_PARAMS
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vcl import VCLNodeDriver as VCL
class VCLTests(unittest.TestCase):

    def setUp(self):
        VCL.connectionCls.conn_class = VCLMockHttp
        VCLMockHttp.type = None
        self.driver = VCL(*VCL_PARAMS)

    def test_list_nodes(self):
        node = self.driver.list_nodes(ipaddr='192.168.1.1')[0]
        self.assertEqual(node.name, 'CentOS 5.4 Base (32 bit VM)')
        self.assertEqual(node.state, NodeState.RUNNING)
        self.assertEqual(node.extra['pass'], 'ehkNGW')

    def test_list_images(self):
        images = self.driver.list_images()
        image = images[0]
        self.assertEqual(image.id, '8')

    def test_list_sizes(self):
        sizes = self.driver.list_sizes()
        self.assertEqual(len(sizes), 1)

    def test_create_node(self):
        image = self.driver.list_images()[0]
        node = self.driver.create_node(image=image)
        self.assertEqual(node.id, '51')

    def test_destroy_node(self):
        node = self.driver.list_nodes(ipaddr='192.168.1.1')[0]
        self.assertTrue(self.driver.destroy_node(node))

    def test_ex_update_node_access(self):
        node = self.driver.list_nodes(ipaddr='192.168.1.1')[0]
        node = self.driver.ex_update_node_access(node, ipaddr='192.168.1.2')
        self.assertEqual(node.name, 'CentOS 5.4 Base (32 bit VM)')
        self.assertEqual(node.state, NodeState.RUNNING)
        self.assertEqual(node.extra['pass'], 'ehkNGW')

    def test_ex_extend_request_time(self):
        node = self.driver.list_nodes(ipaddr='192.168.1.1')[0]
        self.assertTrue(self.driver.ex_extend_request_time(node, 60))

    def test_ex_get_request_end_time(self):
        node = self.driver.list_nodes(ipaddr='192.168.1.1')[0]
        self.assertEqual(self.driver.ex_get_request_end_time(node), 1334168100)