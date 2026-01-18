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
def _1_0_zones(self, method, url, body, headers):
    if method == 'GET':
        if headers['Host'] == 'api.gbt.brightbox.com':
            return self.test_response(httplib.OK, '{}')
        else:
            return self.test_response(httplib.OK, self.fixtures.load('list_zones.json'))