import sys
import unittest
from libcloud.test.secrets import STORAGE_S3_PARAMS
from libcloud.storage.drivers.rgw import (
class S3RGWOutscaleDoubleInstanceTests(S3RGWTests):
    driver_type = S3RGWOutscaleStorageDriver
    default_host = 'osu.eu-west-2.outscale.com'

    def setUp(self):
        self.driver_v2 = self.driver_type(*self.driver_args, signature_version='2')
        self.driver_v4 = self.driver_type(*self.driver_args, signature_version='4')

    def test_connection_class_type(self):
        res = self.driver_v2.connectionCls is S3RGWConnectionAWS2
        self.assertTrue(res, 'driver.connectionCls does not match!')
        res = self.driver_v4.connectionCls is S3RGWConnectionAWS4
        self.assertTrue(res, 'driver.connectionCls does not match!')
        res = self.driver_v2.connectionCls is S3RGWConnectionAWS2
        self.assertTrue(res, 'driver.connectionCls does not match!')

    def test_connection_class_host(self):
        host = self.driver_v2.connectionCls.host
        self.assertEqual(host, self.default_host)
        host = self.driver_v4.connectionCls.host
        self.assertEqual(host, self.default_host)