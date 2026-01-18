import time
from tests.unit import unittest
from boto.cloudsearch.layer1 import Layer1
from boto.cloudsearch.layer2 import Layer2
from boto.regioninfo import RegionInfo
class CloudSearchLayer1Test(unittest.TestCase):
    cloudsearch = True

    def setUp(self):
        super(CloudSearchLayer1Test, self).setUp()
        self.layer1 = Layer1()
        self.domain_name = 'test-%d' % int(time.time())

    def test_create_domain(self):
        resp = self.layer1.create_domain(self.domain_name)
        self.addCleanup(self.layer1.delete_domain, self.domain_name)
        self.assertTrue(resp.get('created', False))