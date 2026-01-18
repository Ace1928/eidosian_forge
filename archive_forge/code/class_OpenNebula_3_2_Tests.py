import sys
import unittest
import libcloud.compute.drivers.opennebula
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeState
from libcloud.test.secrets import OPENNEBULA_PARAMS
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.opennebula import (
class OpenNebula_3_2_Tests(unittest.TestCase):
    """
    OpenNebula.org test suite for OpenNebula v3.2.
    """

    def setUp(self):
        """
        Setup test environment.
        """
        OpenNebulaNodeDriver.connectionCls.conn_class = OpenNebula_3_2_MockHttp
        self.driver = OpenNebulaNodeDriver(*OPENNEBULA_PARAMS + ('3.2',), host='dummy')

    def test_reboot_node(self):
        """
        Test reboot_node functionality.
        """
        image = NodeImage(id=5, name='Ubuntu 9.04 LAMP', driver=self.driver)
        node = Node(5, None, None, None, None, self.driver, image=image)
        ret = self.driver.reboot_node(node)
        self.assertTrue(ret)

    def test_list_sizes(self):
        """
        Test ex_list_networks functionality.
        """
        sizes = self.driver.list_sizes()
        self.assertEqual(len(sizes), 3)
        size = sizes[0]
        self.assertEqual(size.id, '1')
        self.assertEqual(size.name, 'small')
        self.assertEqual(size.ram, 1024)
        self.assertTrue(size.cpu is None or isinstance(size.cpu, float))
        self.assertTrue(size.vcpu is None or isinstance(size.vcpu, int))
        self.assertEqual(size.cpu, 1)
        self.assertIsNone(size.vcpu)
        self.assertIsNone(size.disk)
        self.assertIsNone(size.bandwidth)
        self.assertIsNone(size.price)
        size = sizes[1]
        self.assertEqual(size.id, '2')
        self.assertEqual(size.name, 'medium')
        self.assertEqual(size.ram, 4096)
        self.assertTrue(size.cpu is None or isinstance(size.cpu, float))
        self.assertTrue(size.vcpu is None or isinstance(size.vcpu, int))
        self.assertEqual(size.cpu, 4)
        self.assertIsNone(size.vcpu)
        self.assertIsNone(size.disk)
        self.assertIsNone(size.bandwidth)
        self.assertIsNone(size.price)
        size = sizes[2]
        self.assertEqual(size.id, '3')
        self.assertEqual(size.name, 'large')
        self.assertEqual(size.ram, 8192)
        self.assertTrue(size.cpu is None or isinstance(size.cpu, float))
        self.assertTrue(size.vcpu is None or isinstance(size.vcpu, int))
        self.assertEqual(size.cpu, 8)
        self.assertIsNone(size.vcpu)
        self.assertIsNone(size.disk)
        self.assertIsNone(size.bandwidth)
        self.assertIsNone(size.price)