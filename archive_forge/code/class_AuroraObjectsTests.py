import sys
import unittest
from libcloud.common.types import LibcloudError
from libcloud.test.storage.test_s3 import S3Tests, S3MockHttp
from libcloud.storage.drivers.auroraobjects import AuroraObjectsStorageDriver
class AuroraObjectsTests(S3Tests, unittest.TestCase):
    driver_type = AuroraObjectsStorageDriver

    def setUp(self):
        super().setUp()
        AuroraObjectsStorageDriver.connectionCls.conn_class = S3MockHttp
        S3MockHttp.type = None
        self.driver = self.create_driver()

    def test_get_object_cdn_url(self):
        self.mock_response_klass.type = 'get_object'
        obj = self.driver.get_object(container_name='test2', object_name='test')
        with self.assertRaises(LibcloudError):
            self.driver.get_object_cdn_url(obj)