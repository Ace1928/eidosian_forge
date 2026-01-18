import sys
import unittest
from libcloud.test import LibcloudTestCase
from libcloud.common.types import LibcloudError
from libcloud.storage.base import Object, Container
from libcloud.test.secrets import STORAGE_S3_PARAMS
from libcloud.storage.drivers.digitalocean_spaces import (
class DigitalOceanSpacesTests_v4(DigitalOceanSpacesTests):
    driver_type = DigitalOceanSpacesStorageDriver
    driver_args = STORAGE_S3_PARAMS
    default_host = 'nyc3.digitaloceanspaces.com'

    @classmethod
    def create_driver(self):
        return self.driver_type(*self.driver_args, signature_version='4')

    def test_connection_class_type(self):
        res = self.driver.connectionCls is DOSpacesConnectionAWS4
        self.assertTrue(res, 'driver.connectionCls does not match!')

    def test_connection_class_host(self):
        host = self.driver.connectionCls.host
        self.assertEqual(host, self.default_host)

    def test_valid_regions(self):
        for region, hostname in DO_SPACES_HOSTS_BY_REGION.items():
            driver = self.driver_type(*self.driver_args, region=region)
            self.assertEqual(driver.connectionCls.host, hostname)
            self.assertTrue(driver.connectionCls.host.startswith(region))