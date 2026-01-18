import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import (
def _api_2_0_servers(self, method, url, body, headers):
    if method == 'POST':
        parsed = json.loads(body)
        if 'vlan' in parsed['name']:
            self.assertEqual(len(parsed['nics']), 2)
            body = self.fixtures.load('servers_create_with_vlan.json')
        else:
            body = self.fixtures.load('servers_create.json')
        return (httplib.CREATED, body, {}, httplib.responses[httplib.CREATED])