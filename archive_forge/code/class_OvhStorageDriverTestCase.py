import sys
import unittest
from libcloud.test.secrets import STORAGE_S3_PARAMS
from libcloud.storage.drivers.s3 import S3SignatureV4Connection
from libcloud.storage.drivers.ovh import OVH_FR_SBG_HOST, OvhStorageDriver
from libcloud.test.storage.test_s3 import S3Tests, S3MockHttp
class OvhStorageDriverTestCase(S3Tests, unittest.TestCase):
    driver_type = OvhStorageDriver
    driver_args = STORAGE_S3_PARAMS
    default_host = OVH_FR_SBG_HOST

    @classmethod
    def create_driver(self):
        return self.driver_type(*self.driver_args, host=self.default_host)

    def setUp(self):
        super().setUp()
        OvhStorageDriver.connectionCls.conn_class = S3MockHttp
        S3MockHttp.type = None
        self.driver = self.create_driver()

    def test_connection_class_type(self):
        self.assertEqual(self.driver.connectionCls, S3SignatureV4Connection)

    def test_connection_class_default_host(self):
        self.assertEqual(self.driver.connectionCls.host, self.default_host)
        self.assertEqual(self.driver.connectionCls.port, 443)
        self.assertEqual(self.driver.connectionCls.secure, True)