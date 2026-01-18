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
def _1_0_cloud_ips_cip_jsjc5(self, method, url, body, headers):
    if method == 'DELETE':
        return self.test_response(httplib.OK, '')
    elif method == 'PUT':
        body = json.loads(body)
        if body.get('reverse_dns', None) == 'fred.co.uk':
            return self.test_response(httplib.OK, '')
        else:
            return self.test_response(httplib.BAD_REQUEST, '{"error_name":"bad dns", "errors": ["Bad dns"]}')