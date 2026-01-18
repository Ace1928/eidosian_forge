import sys
import unittest
import libcloud.compute.drivers.opennebula
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeState
from libcloud.test.secrets import OPENNEBULA_PARAMS
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.opennebula import (
class OpenNebula_3_0_Tests(unittest.TestCase):
    """
    OpenNebula.org test suite for OpenNebula v3.0.
    """

    def setUp(self):
        """
        Setup test environment.
        """
        OpenNebulaNodeDriver.connectionCls.conn_class = OpenNebula_3_0_MockHttp
        self.driver = OpenNebulaNodeDriver(*OPENNEBULA_PARAMS + ('3.0',), host='dummy')

    def test_ex_list_networks(self):
        """
        Test ex_list_networks functionality.
        """
        networks = self.driver.ex_list_networks()
        self.assertEqual(len(networks), 2)
        network = networks[0]
        self.assertEqual(network.id, '5')
        self.assertEqual(network.name, 'Network 5')
        self.assertEqual(network.address, '192.168.0.0')
        self.assertEqual(network.size, '256')
        self.assertEqual(network.extra['public'], 'YES')
        network = networks[1]
        self.assertEqual(network.id, '15')
        self.assertEqual(network.name, 'Network 15')
        self.assertEqual(network.address, '192.168.1.0')
        self.assertEqual(network.size, '256')
        self.assertEqual(network.extra['public'], 'NO')

    def test_ex_node_set_save_name(self):
        """
        Test ex_node_action functionality.
        """
        image = NodeImage(id=5, name='Ubuntu 9.04 LAMP', driver=self.driver)
        node = Node(5, None, None, None, None, self.driver, image=image)
        ret = self.driver.ex_node_set_save_name(node, 'test')
        self.assertTrue(ret)