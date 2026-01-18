import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.kubevirt import KubeVirtNodeDriver
from libcloud.test.common.test_kubernetes import KubernetesAuthTestCaseMixin
class KubeVirtTestCase(unittest.TestCase, KubernetesAuthTestCaseMixin):
    driver_cls = KubeVirtNodeDriver
    fixtures = ComputeFileFixtures('kubevirt')

    def setUp(self):
        KubeVirtNodeDriver.connectionCls.conn_class = KubeVirtMockHttp
        self.driver = KubeVirtNodeDriver(key='user', secret='pass', secure=True, host='foo', port=6443)

    def test_list_locations(self):
        locations = self.driver.list_locations()
        self.assertEqual(len(locations), 5)
        self.assertEqual(locations[0].name, 'default')
        self.assertEqual(locations[1].name, 'kube-node-lease')
        self.assertEqual(locations[2].name, 'kube-public')
        self.assertEqual(locations[3].name, 'kube-system')
        namespace4 = locations[0].driver.list_locations()[4].name
        self.assertEqual(namespace4, 'kubevirt')
        id4 = locations[2].driver.list_locations()[4].id
        self.assertEqual(id4, 'e6d3d7e8-0ee5-428b-8e17-5187779e5627')

    def test_list_nodes(self):
        nodes = self.driver.list_nodes()
        id0 = '74fd7665-fbd6-4565-977c-96bd21fb785a'
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].extra['namespace'], 'default')
        valid_node_states = {NodeState.RUNNING, NodeState.PENDING, NodeState.STOPPED}
        self.assertTrue(nodes[0].state in valid_node_states)
        self.assertEqual(nodes[0].name, 'testvm')
        self.assertEqual(nodes[0].id, id0)

    def test_destroy_node(self):
        nodes = self.driver.list_nodes()
        to_destroy = nodes[-1]
        resp = self.driver.destroy_node(to_destroy)
        self.assertTrue(resp)

    def test_start_node(self):
        nodes = self.driver.list_nodes()
        r1 = self.driver.start_node(nodes[0])
        self.assertTrue(r1)

    def test_stop_node(self):
        nodes = self.driver.list_nodes()
        r1 = self.driver.stop_node(nodes[0])
        self.assertTrue(r1)

    def test_reboot_node(self):
        nodes = self.driver.list_nodes()
        for node in nodes:
            if node.name == 'testvm':
                resp = self.driver.reboot_node(node)
        self.assertTrue(resp)