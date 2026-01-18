import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import (
def _api_2_0_keypairs_186106ac_afb5_40e5_a0de_6f0feba5a3d5(self, method, url, body, headers):
    if method == 'GET':
        body = self.fixtures.load('keypairs_get.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])
    if method == 'DELETE':
        body = ''
        return (httplib.NO_CONTENT, body, {}, httplib.responses[httplib.NO_CONTENT])