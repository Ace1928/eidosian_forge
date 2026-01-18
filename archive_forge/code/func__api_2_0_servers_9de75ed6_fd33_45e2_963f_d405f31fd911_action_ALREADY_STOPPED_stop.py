import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import (
def _api_2_0_servers_9de75ed6_fd33_45e2_963f_d405f31fd911_action_ALREADY_STOPPED_stop(self, method, url, body, headers):
    body = self.fixtures.load('stop_already_stopped.json')
    return (httplib.FORBIDDEN, body, {}, httplib.responses[httplib.FORBIDDEN])