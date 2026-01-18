import sys
import unittest
from datetime import datetime
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.compute.base import NodeSize
from libcloud.test.secrets import GRIDSCALE_PARAMS
from libcloud.utils.iso8601 import UTC
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.gridscale import GridscaleNodeDriver
class Gridscale_Tests(LibcloudTestCase):

    def setUp(self):
        GridscaleNodeDriver.connectionCls.conn_class = GridscaleMockHttp
        GridscaleMockHttp.type = None
        self.driver = GridscaleNodeDriver(*GRIDSCALE_PARAMS)

    def test_create_node_success(self):
        image = self.driver.list_images()[0]
        size = NodeSize(id=0, name='test', bandwidth=0, price=0, ram=10240, driver=self.driver, disk=50, extra={'cores': 2})
        location = self.driver.list_locations()[0]
        sshkey = ['b1682d3a-1869-4bdc-8ffe-e74a261d300c']
        GridscaleMockHttp.type = 'POST'
        node = self.driver.create_node(name='test', size=size, image=image, location=location, ex_ssh_key_ids=sshkey)
        self.assertEqual(node.name, 'test.test')
        self.assertEqual(node.public_ips, ['185.102.95.236', '2a06:2380:0:1::211'])

    def test_create_image_success(self):
        node = self.driver.list_nodes()[0]
        GridscaleMockHttp.type = 'POST'
        image = self.driver.create_image(node, 'test.test')
        self.assertTrue(image)

    def test_list_nodes_success(self):
        nodes = self.driver.list_nodes()
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].name, 'test.test')
        self.assertEqual(nodes[0].public_ips, ['185.102.95.236', '2a06:2380:0:1::211'])
        self.assertEqual(nodes[0].extra['cores'], 2)
        self.assertEqual(nodes[0].extra['memory'], 10240)

    def test_list_locations_success(self):
        locations = self.driver.list_locations()
        self.assertTrue(len(locations) >= 1)

    def test_list_volumes(self):
        volumes = self.driver.list_volumes()
        self.assertEqual(len(volumes), 1)
        volume = volumes[0]
        self.assertEqual(volume.id, 'e66bb753-4a03-4ee2-a069-a601f393c9ee')
        self.assertEqual(volume.name, 'linux')
        self.assertEqual(volume.size, 50)
        self.assertEqual(volume.driver, self.driver)

    def test_list_images_success(self):
        images = self.driver.list_images()
        self.assertTrue(len(images) >= 1)
        image = images[0]
        self.assertTrue(image.id is not None)
        self.assertTrue(image.name is not None)

    def test_list_key_pairs(self):
        keys = self.driver.list_key_pairs()
        self.assertEqual(len(keys), 2)
        self.assertEqual(keys[0].name, 'karl')
        self.assertEqual(keys[0].public_key, 'ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC')

    def test_list_volume_snapshots(self):
        volume = self.driver.list_volumes()[0]
        snapshots = self.driver.list_volume_snapshots(volume)
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0].id, 'd755de62-4d75-4d61-addd-a5c9743a5deb')

    def test_list_volumes_empty(self):
        GridscaleMockHttp.type = 'EMPTY'
        volumes = self.driver.list_volumes()
        self.assertEqual(len(volumes), 0)

    def test_ex_list_networks(self):
        networks = self.driver.ex_list_networks()[0]
        self.assertEqual(networks.id, '1196529b-a8de-417f')

    def test_ex_list_ips(self):
        ip = self.driver.ex_list_ips()[0]
        self.assertEqual(ip.id, '56b8d161-325b-4fd4')

    def test_ex_destroy_ip(self):
        ip = self.driver.ex_list_ips()[0]
        GridscaleMockHttp.type = 'DELETE'
        self.assertTrue(self.driver.ex_destroy_ip(ip))

    def test_ex_destroy_network(self):
        network = self.driver.ex_list_networks()[0]
        GridscaleMockHttp.type = 'DELETE'
        self.assertTrue(self.driver.ex_destroy_network(network))

    def test_destroy_node_success(self):
        node = self.driver.list_nodes()[0]
        GridscaleMockHttp.type = 'DELETE'
        res = self.driver.destroy_node(node)
        self.assertTrue(res)
        res = self.driver.destroy_node(node, ex_destroy_associated_resources=True)
        self.assertTrue(res)

    def test_destroy_volume(self):
        volume = self.driver.list_volumes()[0]
        GridscaleMockHttp.type = 'DELETE'
        res = self.driver.destroy_volume(volume)
        self.assertTrue(res)

    def test_destroy_volume_snapshot(self):
        volume = self.driver.list_volumes()[0]
        snapshot = self.driver.list_volume_snapshots(volume)[0]
        GridscaleMockHttp.type = 'DELETE'
        res = self.driver.destroy_volume_snapshot(snapshot)
        self.assertTrue(res)

    def test_get_image_success(self):
        image = self.driver.get_image('12345')
        self.assertEqual(image.id, '12345')

    def test_list_nodes_fills_created_datetime(self):
        nodes = self.driver.list_nodes()
        self.assertEqual(nodes[0].created_at, datetime(2019, 6, 7, 12, 56, 44, tzinfo=UTC))

    def test_ex_list_volumes_for_node(self):
        node = self.driver.list_nodes()[0]
        volumes = self.driver.ex_list_volumes_for_node(node=node)
        self.assertEqual(len(volumes), 1)
        self.assertEqual(volumes[0].size, 50)

    def test_ex_list_ips_for_node(self):
        node = self.driver.list_nodes()[0]
        ips = self.driver.ex_list_ips_for_node(node=node)
        self.assertEqual(len(ips), 1)
        self.assertEqual(ips[0].ip_address, '185.102.95.236')

    def test_ex_rename_node(self):
        node = self.driver.list_nodes()[0]
        self.assertTrue(self.driver.ex_rename_node(node, name='new-name'))

    def test_ex_rename_volume(self):
        volume = self.driver.list_volumes()[0]
        self.assertTrue(self.driver.ex_rename_volume(volume, name='new-name'))

    def test_ex_network(self):
        network = self.driver.ex_list_networks()[0]
        self.assertTrue(self.driver.ex_rename_network(network, name='new-name'))