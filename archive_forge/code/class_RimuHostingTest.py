import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.test.compute import TestCaseMixin
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.rimuhosting import RimuHostingNodeDriver
class RimuHostingTest(unittest.TestCase, TestCaseMixin):

    def setUp(self):
        RimuHostingNodeDriver.connectionCls.conn_class = RimuHostingMockHttp
        self.driver = RimuHostingNodeDriver('foo')

    def test_list_nodes(self):
        nodes = self.driver.list_nodes()
        self.assertEqual(len(nodes), 1)
        node = nodes[0]
        self.assertEqual(node.public_ips[0], '1.2.3.4')
        self.assertEqual(node.public_ips[1], '1.2.3.5')
        self.assertEqual(node.extra['order_oid'], 88833465)
        self.assertEqual(node.id, 'order-88833465-api-ivan-net-nz')

    def test_list_sizes(self):
        sizes = self.driver.list_sizes()
        self.assertEqual(len(sizes), 1)
        size = sizes[0]
        self.assertEqual(size.ram, 950)
        self.assertEqual(size.disk, 20)
        self.assertEqual(size.bandwidth, 75)
        self.assertEqual(size.price, 32.54)

    def test_list_images(self):
        images = self.driver.list_images()
        self.assertEqual(len(images), 6)
        image = images[0]
        self.assertEqual(image.name, 'Debian 5.0 (aka Lenny, RimuHosting recommended distro)')
        self.assertEqual(image.id, 'lenny')

    def test_reboot_node(self):
        node = self.driver.list_nodes()[0]
        self.driver.reboot_node(node)

    def test_destroy_node(self):
        node = self.driver.list_nodes()[0]
        self.driver.destroy_node(node)

    def test_create_node(self):
        size = self.driver.list_sizes()[0]
        image = self.driver.list_images()[0]
        self.driver.create_node(name='api.ivan.net.nz', image=image, size=size)