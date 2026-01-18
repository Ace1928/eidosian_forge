import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import (
def _api_2_0_fwpolicies_0e339282_0cb5_41ac_a9db_727fb62ff2dc(self, method, url, body, headers):
    if method == 'DELETE':
        body = ''
        return (httplib.NO_CONTENT, body, {}, httplib.responses[httplib.NO_CONTENT])