import base64
import unittest
from libcloud.utils.py3 import b
from libcloud.test.storage.test_atmos import AtmosTests, AtmosMockHttp
from libcloud.storage.drivers.ninefold import NinefoldStorageDriver
class NinefoldTests(AtmosTests, unittest.TestCase):

    def setUp(self):
        NinefoldStorageDriver.connectionCls.conn_class = AtmosMockHttp
        NinefoldStorageDriver.path = ''
        AtmosMockHttp.type = None
        AtmosMockHttp.upload_created = False
        self.driver = NinefoldStorageDriver('dummy', base64.b64encode(b('dummy')))
        self._remove_test_file()