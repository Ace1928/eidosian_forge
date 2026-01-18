import sys
import unittest
from datetime import datetime
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.compute.base import NodeImage
from libcloud.test.secrets import SCALEWAY_PARAMS
from libcloud.utils.iso8601 import UTC
from libcloud.common.exceptions import BaseHTTPError
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.scaleway import ScalewayNodeDriver
class Scaleway_Tests(LibcloudTestCase):

    def setUp(self):
        ScalewayNodeDriver.connectionCls.conn_class = ScalewayMockHttp
        ScalewayMockHttp.type = None
        self.driver = ScalewayNodeDriver(*SCALEWAY_PARAMS)

    def test_authentication(self):
        ScalewayMockHttp.type = 'UNAUTHORIZED'
        assertRaisesRegex(self, BaseHTTPError, 'Authentication error', self.driver.list_nodes)

    def test_list_locations_success(self):
        locations = self.driver.list_locations()
        self.assertTrue(len(locations) >= 1)
        location = locations[0]
        self.assertEqual(location.id, 'par1')
        self.assertEqual(location.name, 'Paris 1')

    def test_list_sizes_success(self):
        sizes = self.driver.list_sizes()
        self.assertTrue(len(sizes) >= 1)
        size = sizes[0]
        self.assertTrue(size.id is not None)
        self.assertEqual(size.name, 'ARM64-4GB')
        self.assertEqual(size.ram, 4096)
        size = sizes[1]
        self.assertTrue(size.id is not None)
        self.assertEqual(size.name, 'START1-XS')
        self.assertEqual(size.ram, 1024)
        size = sizes[2]
        self.assertTrue(size.id is not None)
        self.assertEqual(size.name, 'X64-120GB')
        self.assertEqual(size.ram, 122880)

    def test_list_images_success(self):
        images = self.driver.list_images()
        self.assertTrue(len(images) >= 1)
        image = images[0]
        self.assertTrue(image.id is not None)
        self.assertTrue(image.name is not None)

    def test_create_image_success(self):
        node = self.driver.list_nodes()[0]
        ScalewayMockHttp.type = 'POST'
        image = self.driver.create_image(node, 'my_image')
        self.assertEqual(image.name, 'my_image')
        self.assertEqual(image.id, '98bf3ac2-a1f5-471d-8c8f-1b706ab57ef0')
        self.assertEqual(image.extra['arch'], 'arm')

    def test_delete_image_success(self):
        image = self.driver.get_image(12345)
        ScalewayMockHttp.type = 'DELETE'
        result = self.driver.delete_image(image)
        self.assertTrue(result)

    def test_get_image_success(self):
        image = self.driver.get_image(12345)
        self.assertEqual(image.name, 'my_image')
        self.assertEqual(image.id, '12345')
        self.assertEqual(image.extra['arch'], 'arm')

    def test_list_nodes_success(self):
        nodes = self.driver.list_nodes()
        self.assertEqual(len(nodes), 2)
        self.assertEqual(nodes[0].name, 'my_server')
        self.assertEqual(nodes[0].public_ips, [])
        self.assertEqual(nodes[0].extra['volumes']['0']['id'], 'c1eb8f3a-4f0b-4b95-a71c-93223e457f5a')
        self.assertEqual(nodes[0].extra['organization'], '000a115d-2852-4b0a-9ce8-47f1134ba95a')

    def test_list_nodes_fills_created_datetime(self):
        nodes = self.driver.list_nodes()
        self.assertEqual(nodes[0].created_at, datetime(2014, 5, 22, 12, 57, 22, 514298, tzinfo=UTC))

    def test_create_node_success(self):
        image = self.driver.list_images()[0]
        size = self.driver.list_sizes()[0]
        location = self.driver.list_locations()[0]
        ScalewayMockHttp.type = 'POST'
        node = self.driver.create_node(name='test', size=size, image=image, region=location)
        self.assertEqual(node.name, 'my_server')
        self.assertEqual(node.public_ips, [])
        self.assertEqual(node.extra['volumes']['0']['id'], 'd9257116-6919-49b4-a420-dcfdff51fcb1')
        self.assertEqual(node.extra['organization'], '000a115d-2852-4b0a-9ce8-47f1134ba95a')

    def test_create_node_invalid_size(self):
        image = NodeImage(id='01234567-89ab-cdef-fedc-ba9876543210', name=None, driver=self.driver)
        size = self.driver.list_sizes()[0]
        location = self.driver.list_locations()[0]
        ScalewayMockHttp.type = 'INVALID_IMAGE'
        expected_msg = '" not found'
        assertRaisesRegex(self, Exception, expected_msg, self.driver.create_node, name='test', size=size, image=image, region=location)

    def test_reboot_node_success(self):
        node = self.driver.list_nodes()[0]
        ScalewayMockHttp.type = 'REBOOT'
        result = self.driver.reboot_node(node)
        self.assertTrue(result)

    def test_destroy_node_success(self):
        node = self.driver.list_nodes()[0]
        ScalewayMockHttp.type = 'TERMINATE'
        result = self.driver.destroy_node(node)
        self.assertTrue(result)

    def test_list_volumes(self):
        volumes = self.driver.list_volumes()
        self.assertEqual(len(volumes), 2)
        volume = volumes[0]
        self.assertEqual(volume.id, 'f929fe39-63f8-4be8-a80e-1e9c8ae22a76')
        self.assertEqual(volume.name, 'volume-0-1')
        self.assertEqual(volume.size, 10)
        self.assertEqual(volume.driver, self.driver)

    def test_list_volumes_empty(self):
        ScalewayMockHttp.type = 'EMPTY'
        volumes = self.driver.list_volumes()
        self.assertEqual(len(volumes), 0)

    def test_list_volume_snapshots(self):
        volume = self.driver.list_volumes()[0]
        snapshots = self.driver.list_volume_snapshots(volume)
        self.assertEqual(len(snapshots), 2)
        snapshot1, snapshot2 = snapshots
        self.assertEqual(snapshot1.id, '6f418e5f-b42d-4423-a0b5-349c74c454a4')
        self.assertEqual(snapshot2.id, 'c6ff5501-eb35-44b8-aa01-8777211a830b')

    def test_create_volume(self):
        par1 = [r for r in self.driver.list_locations() if r.id == 'par1'][0]
        ScalewayMockHttp.type = 'POST'
        volume = self.driver.create_volume(10, 'volume-0-3', par1)
        self.assertEqual(volume.id, 'c675f420-cfeb-48ff-ba2a-9d2a4dbe3fcd')
        self.assertEqual(volume.name, 'volume-0-3')
        self.assertEqual(volume.size, 10)
        self.assertEqual(volume.driver, self.driver)

    def test_create_volume_snapshot(self):
        volume = self.driver.list_volumes()[0]
        ScalewayMockHttp.type = 'POST'
        snapshot = self.driver.create_volume_snapshot(volume, 'snapshot-0-1')
        self.assertEqual(snapshot.id, 'f0361e7b-cbe4-4882-a999-945192b7171b')
        self.assertEqual(snapshot.extra['volume_type'], 'l_ssd')
        self.assertEqual(volume.driver, self.driver)

    def test_destroy_volume(self):
        volume = self.driver.list_volumes()[0]
        ScalewayMockHttp.type = 'DELETE'
        resp = self.driver.destroy_volume(volume)
        self.assertTrue(resp)

    def test_destroy_volume_snapshot(self):
        volume = self.driver.list_volumes()[0]
        snapshot = self.driver.list_volume_snapshots(volume)[0]
        ScalewayMockHttp.type = 'DELETE'
        result = self.driver.destroy_volume_snapshot(snapshot)
        self.assertTrue(result)

    def test_list_key_pairs(self):
        keys = self.driver.list_key_pairs()
        self.assertEqual(len(keys), 1)
        self.assertEqual(keys[0].name, 'example')
        self.assertEqual(keys[0].public_key, 'ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAAAQQDGk5')
        self.assertEqual(keys[0].fingerprint, 'f5:d1:78:ed:28:72:5f:e1:ac:94:fd:1f:e0:a3:48:6d')

    def test_import_key_pair_from_string(self):
        result = self.driver.import_key_pair_from_string(name='example', key_material='ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAAAQQDGk5')
        self.assertTrue(result)

    def test_delete_key_pair(self):
        key = self.driver.list_key_pairs()[0]
        result = self.driver.delete_key_pair(key)
        self.assertTrue(result)