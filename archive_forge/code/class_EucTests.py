import os
import sys
import base64
from datetime import datetime
from collections import OrderedDict
from libcloud.test import MockHttp, LibcloudTestCase, unittest
from libcloud.utils.py3 import b, httplib, parse_qs
from libcloud.compute.base import (
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import EC2_PARAMS
from libcloud.compute.types import (
from libcloud.utils.iso8601 import UTC
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.ec2 import (
class EucTests(LibcloudTestCase, TestCaseMixin):

    def setUp(self):
        EucNodeDriver.connectionCls.conn_class = EucMockHttp
        EC2MockHttp.use_param = 'Action'
        EC2MockHttp.type = None
        self.driver = EucNodeDriver(key=EC2_PARAMS[0], secret=EC2_PARAMS[1], host='some.eucalyptus.com', api_version='3.4.1')

    def test_list_locations_response(self):
        try:
            self.driver.list_locations()
        except Exception:
            pass
        else:
            self.fail('Exception was not thrown')

    def test_list_location(self):
        pass

    def test_list_sizes(self):
        sizes = self.driver.list_sizes()
        ids = [s.id for s in sizes]
        self.assertEqual(len(ids), 18)
        self.assertTrue('t1.micro' in ids)
        self.assertTrue('m1.medium' in ids)
        self.assertTrue('m3.xlarge' in ids)