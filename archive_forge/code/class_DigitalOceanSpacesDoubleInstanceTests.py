import sys
import unittest
from libcloud.test import LibcloudTestCase
from libcloud.common.types import LibcloudError
from libcloud.storage.base import Object, Container
from libcloud.test.secrets import STORAGE_S3_PARAMS
from libcloud.storage.drivers.digitalocean_spaces import (
class DigitalOceanSpacesDoubleInstanceTests(LibcloudTestCase):
    driver_type = DigitalOceanSpacesStorageDriver
    driver_args = STORAGE_S3_PARAMS
    default_host = 'nyc3.digitaloceanspaces.com'
    alt_host = 'ams3.digitaloceanspaces.com'

    def setUp(self):
        self.driver_v2 = self.driver_type(*self.driver_args, signature_version='2')
        self.driver_v4 = self.driver_type(*self.driver_args, signature_version='4', region='ams3')

    def test_connection_class_type(self):
        res = self.driver_v2.connectionCls is DOSpacesConnectionAWS2
        self.assertTrue(res, 'driver.connectionCls does not match!')
        res = self.driver_v4.connectionCls is DOSpacesConnectionAWS4
        self.assertTrue(res, 'driver.connectionCls does not match!')
        res = self.driver_v2.connectionCls is DOSpacesConnectionAWS2
        self.assertTrue(res, 'driver.connectionCls does not match!')

    def test_connection_class_host(self):
        host = self.driver_v2.connectionCls.host
        self.assertEqual(host, self.default_host)
        host = self.driver_v4.connectionCls.host
        self.assertEqual(host, self.alt_host)