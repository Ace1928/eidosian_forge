import sys
import unittest
import libcloud.compute.drivers.opennebula
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeState
from libcloud.test.secrets import OPENNEBULA_PARAMS
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.opennebula import (
class OpenNebula_3_6_Tests(unittest.TestCase):
    """
    OpenNebula.org test suite for OpenNebula v3.6.
    """

    def setUp(self):
        """
        Setup test environment.
        """
        OpenNebulaNodeDriver.connectionCls.conn_class = OpenNebula_3_6_MockHttp
        self.driver = OpenNebulaNodeDriver(*OPENNEBULA_PARAMS + ('3.6',), host='dummy')

    def test_create_volume(self):
        new_volume = self.driver.create_volume(1000, 'test-volume')
        self.assertEqual(new_volume.id, '5')
        self.assertEqual(new_volume.size, 1000)
        self.assertEqual(new_volume.name, 'test-volume')

    def test_destroy_volume(self):
        images = self.driver.list_images()
        self.assertEqual(len(images), 2)
        image = images[0]
        ret = self.driver.destroy_volume(image)
        self.assertTrue(ret)

    def test_attach_volume(self):
        nodes = self.driver.list_nodes()
        node = nodes[0]
        images = self.driver.list_images()
        image = images[0]
        ret = self.driver.attach_volume(node, image, 'sda')
        self.assertTrue(ret)

    def test_detach_volume(self):
        images = self.driver.list_images()
        image = images[1]
        ret = self.driver.detach_volume(image)
        self.assertTrue(ret)
        nodes = self.driver.list_nodes()
        node = nodes[1]
        ret = self.driver.detach_volume(node.image)
        self.assertFalse(ret)

    def test_list_volumes(self):
        volumes = self.driver.list_volumes()
        self.assertEqual(len(volumes), 2)
        volume = volumes[0]
        self.assertEqual(volume.id, '5')
        self.assertEqual(volume.size, 2048)
        self.assertEqual(volume.name, 'Ubuntu 9.04 LAMP')
        volume = volumes[1]
        self.assertEqual(volume.id, '15')
        self.assertEqual(volume.size, 1024)
        self.assertEqual(volume.name, 'Debian Sid')