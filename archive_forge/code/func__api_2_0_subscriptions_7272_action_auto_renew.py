import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import (
def _api_2_0_subscriptions_7272_action_auto_renew(self, method, url, body, headers):
    body = ''
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])