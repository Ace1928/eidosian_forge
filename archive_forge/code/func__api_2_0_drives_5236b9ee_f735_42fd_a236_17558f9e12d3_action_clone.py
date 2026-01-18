import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import (
def _api_2_0_drives_5236b9ee_f735_42fd_a236_17558f9e12d3_action_clone(self, method, url, body, headers):
    body = self.fixtures.load('drives_clone.json')
    return (httplib.ACCEPTED, body, {}, httplib.responses[httplib.ACCEPTED])