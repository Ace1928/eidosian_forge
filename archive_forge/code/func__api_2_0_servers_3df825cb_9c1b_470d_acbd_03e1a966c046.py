import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import (
def _api_2_0_servers_3df825cb_9c1b_470d_acbd_03e1a966c046(self, method, url, body, headers):
    if method == 'GET':
        body = self.fixtures.load('servers_get.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])
    elif method == 'PUT':
        body = ''
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])