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
def _ex_user_data_RunInstances(self, method, url, body, headers):
    if url.startswith('/'):
        url = url[1:]
    if url.startswith('?'):
        url = url[1:]
    params = parse_qs(url)
    self.assertTrue('UserData' in params, 'UserData not in params, actual params: %s' % str(params))
    user_data = base64.b64decode(b(params['UserData'][0])).decode('utf-8')
    self.assertEqual(user_data, 'foo\nbar\x0coo')
    body = self.fixtures.load('run_instances.xml')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])