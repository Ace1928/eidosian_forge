import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import (
def _api_2_0_tags_WITH_RESOURCES(self, method, url, body, headers):
    if method == 'POST':
        body = self.fixtures.load('tags_create_with_resources.json')
        return (httplib.CREATED, body, {}, httplib.responses[httplib.CREATED])