import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import (
def _api_2_0_subscriptions_RESOURCE_FILTER(self, method, url, body, headers):
    expected_params = {'resource': 'cpu,mem', 'status': 'all'}
    self.assertUrlContainsQueryParams(url, expected_params)
    body = self.fixtures.load('subscriptions.json')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])