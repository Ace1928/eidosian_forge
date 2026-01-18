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
def _2_0_zones(self, method, url, body, headers):
    data = '{"error_name":"unrecognised_endpoint", "errors": ["The request was for an unrecognised API endpoint"]}'
    return self.test_response(httplib.BAD_REQUEST, data)