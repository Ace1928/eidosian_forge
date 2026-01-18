import sys
import base64
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import b, httplib
from libcloud.common.types import InvalidCredsError
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import BRIGHTBOX_PARAMS
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.brightbox import BrightboxNodeDriver
def _1_0_cloud_ips_cip_jsjc5_map(self, method, url, body, headers):
    if method == 'POST':
        body = json.loads(body)
        if 'destination' in body:
            return self.test_response(httplib.ACCEPTED, '')
        else:
            data = '{"error_name":"bad destination", "errors": ["Bad destination"]}'
            return self.test_response(httplib.BAD_REQUEST, data)