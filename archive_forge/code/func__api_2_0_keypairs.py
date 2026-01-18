import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import (
def _api_2_0_keypairs(self, method, url, body, headers):
    if method == 'GET':
        body = self.fixtures.load('keypairs_list.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])
    elif method == 'POST':
        body = self.fixtures.load('keypairs_import.json')
        return (httplib.CREATED, body, {}, httplib.responses[httplib.CREATED])