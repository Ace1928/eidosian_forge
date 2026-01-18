import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.utils.misc import dict2str, str2list, str2dicts
from libcloud.compute.base import Node
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import CloudSigmaNodeDriver, CloudSigmaZrhNodeDriver
class CloudSigmaAPI10BaseTestCase:
    should_list_locations = False
    driver_klass = CloudSigmaZrhNodeDriver
    driver_kwargs = {}

    def setUp(self):
        self.driver_klass.connectionCls.conn_class = CloudSigmaHttp
        CloudSigmaHttp.type = None
        self.driver = self.driver_klass(*self.driver_args, **self.driver_kwargs)

    def test_list_nodes(self):
        nodes = self.driver.list_nodes()
        self.assertTrue(isinstance(nodes, list))
        self.assertEqual(len(nodes), 1)
        node = nodes[0]
        self.assertEqual(node.public_ips[0], '1.2.3.4')
        self.assertEqual(node.extra['smp'], 1)
        self.assertEqual(node.extra['cpu'], 1100)
        self.assertEqual(node.extra['mem'], 640)

    def test_list_sizes(self):
        images = self.driver.list_sizes()
        self.assertEqual(len(images), 10)

    def test_list_images(self):
        sizes = self.driver.list_images()
        self.assertEqual(len(sizes), 10)

    def test_start_node(self):
        nodes = self.driver.list_nodes()
        node = nodes[0]
        self.assertTrue(self.driver.ex_start_node(node))

    def test_shutdown_node(self):
        nodes = self.driver.list_nodes()
        node = nodes[0]
        self.assertTrue(self.driver.ex_stop_node(node))
        self.assertTrue(self.driver.ex_shutdown_node(node))

    def test_reboot_node(self):
        node = self.driver.list_nodes()[0]
        self.assertTrue(self.driver.reboot_node(node))

    def test_destroy_node(self):
        node = self.driver.list_nodes()[0]
        self.assertTrue(self.driver.destroy_node(node))
        self.driver.list_nodes()

    def test_create_node(self):
        size = self.driver.list_sizes()[0]
        image = self.driver.list_images()[0]
        node = self.driver.create_node(name='cloudsigma node', image=image, size=size)
        self.assertTrue(isinstance(node, Node))

    def test_ex_static_ip_list(self):
        ips = self.driver.ex_static_ip_list()
        self.assertEqual(len(ips), 3)

    def test_ex_static_ip_create(self):
        result = self.driver.ex_static_ip_create()
        self.assertEqual(len(result), 2)
        self.assertEqual(len(list(result[0].keys())), 6)
        self.assertEqual(len(list(result[1].keys())), 6)

    def test_ex_static_ip_destroy(self):
        result = self.driver.ex_static_ip_destroy('1.2.3.4')
        self.assertTrue(result)

    def test_ex_drives_list(self):
        result = self.driver.ex_drives_list()
        self.assertEqual(len(result), 2)

    def test_ex_drive_destroy(self):
        result = self.driver.ex_drive_destroy('d18119ce_7afa_474a_9242_e0384b160220')
        self.assertTrue(result)

    def test_ex_set_node_configuration(self):
        node = self.driver.list_nodes()[0]
        result = self.driver.ex_set_node_configuration(node, **{'smp': 2})
        self.assertTrue(result)

    def test_str2dicts(self):
        string = 'mem 1024\ncpu 2200\n\nmem2048\\cpu 1100'
        result = str2dicts(string)
        self.assertEqual(len(result), 2)

    def test_str2list(self):
        string = 'ip 1.2.3.4\nip 1.2.3.5\nip 1.2.3.6'
        result = str2list(string)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], '1.2.3.4')
        self.assertEqual(result[1], '1.2.3.5')
        self.assertEqual(result[2], '1.2.3.6')

    def test_dict2str(self):
        d = {'smp': 5, 'cpu': 2200, 'mem': 1024}
        result = dict2str(d)
        self.assertTrue(len(result) > 0)
        self.assertTrue(result.find('smp 5') >= 0)
        self.assertTrue(result.find('cpu 2200') >= 0)
        self.assertTrue(result.find('mem 1024') >= 0)