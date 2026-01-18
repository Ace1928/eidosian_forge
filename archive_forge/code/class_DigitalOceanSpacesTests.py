import sys
import unittest
from libcloud.test import LibcloudTestCase
from libcloud.common.types import LibcloudError
from libcloud.storage.base import Object, Container
from libcloud.test.secrets import STORAGE_S3_PARAMS
from libcloud.storage.drivers.digitalocean_spaces import (
class DigitalOceanSpacesTests(LibcloudTestCase):
    driver_type = DigitalOceanSpacesStorageDriver
    driver_args = STORAGE_S3_PARAMS
    default_host = 'nyc3.digitaloceanspaces.com'

    @classmethod
    def create_driver(self):
        return self.driver_type(*self.driver_args, signature_version='2', host=self.default_host)

    def setUp(self):
        self.driver = self.create_driver()
        self.container = Container('test-container', {}, self.driver)
        self.object = Object('test-object', 1, 'hash', {}, 'meta_data', self.container, self.driver)

    def test_connection_class_type(self):
        res = self.driver.connectionCls is DOSpacesConnectionAWS2
        self.assertTrue(res, 'driver.connectionCls does not match!')

    def test_connection_class_host(self):
        host = self.driver.connectionCls.host
        self.assertEqual(host, self.default_host)

    def test_container_enable_cdn_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.container.enable_cdn()

    def test_container_get_cdn_url_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.container.get_cdn_url()

    def test_object_enable_cdn_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.object.enable_cdn()

    def test_object_get_cdn_url_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.object.get_cdn_url()

    def test_invalid_signature_version(self):
        with self.assertRaises(ValueError):
            self.driver_type(*self.driver_args, signature_version='3', host=self.default_host)

    def test_invalid_region(self):
        with self.assertRaises(LibcloudError):
            self.driver_type(*self.driver_args, region='atlantis', host=self.default_host)