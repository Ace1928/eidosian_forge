import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import (
def _api_2_0_drives_9d1d2cf3_08c1_462f_8485_f4b073560809_action_clone(self, method, url, body, headers):
    body = self.fixtures.load('drives_clone.json')
    return (httplib.ACCEPTED, body, {}, httplib.responses[httplib.ACCEPTED])