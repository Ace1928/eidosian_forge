import sys
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CLOUDSCALE_PARAMS
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudscale import CloudscaleNodeDriver
class CloudscaleTests(LibcloudTestCase):

    def setUp(self):
        CloudscaleNodeDriver.connectionCls.conn_class = CloudscaleMockHttp
        self.driver = CloudscaleNodeDriver(*CLOUDSCALE_PARAMS)

    def test_list_images_success(self):
        images = self.driver.list_images()
        image, = images
        self.assertTrue(image.id is not None)
        self.assertTrue(image.name is not None)

    def test_list_sizes_success(self):
        sizes = self.driver.list_sizes()
        self.assertEqual(len(sizes), 2)
        size = sizes[0]
        self.assertTrue(size.id is not None)
        self.assertEqual(size.name, 'Flex-2')
        self.assertEqual(size.ram, 2048)
        size = sizes[1]
        self.assertTrue(size.id is not None)
        self.assertEqual(size.name, 'Flex-4')
        self.assertEqual(size.ram, 4096)

    def test_list_locations_not_existing(self):
        try:
            self.driver.list_locations()
        except NotImplementedError:
            pass
        else:
            assert False, 'Did not raise the wished error.'

    def test_list_nodes_success(self):
        nodes = self.driver.list_nodes()
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].id, '47cec963-fcd2-482f-bdb6-24461b2d47b1')
        self.assertEqual(nodes[0].public_ips, ['185.98.122.176', '2a06:c01:1:1902::7ab0:176'])

    def test_reboot_node_success(self):
        node = self.driver.list_nodes()[0]
        result = self.driver.reboot_node(node)
        self.assertTrue(result)

    def test_create_node_success(self):
        test_size = self.driver.list_sizes()[0]
        test_image = self.driver.list_images()[0]
        created_node = self.driver.create_node('node-name', test_size, test_image)
        self.assertEqual(created_node.id, '47cec963-fcd2-482f-bdb6-24461b2d47b1')

    def test_destroy_node_success(self):
        node = self.driver.list_nodes()[0]
        result = self.driver.destroy_node(node)
        self.assertTrue(result)