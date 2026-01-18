import sys
import json
import unittest
import libcloud.compute.drivers.equinixmetal
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, KeyPair
from libcloud.test.compute import TestCaseMixin
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.equinixmetal import EquinixMetalNodeDriver
def _metal_v1_bgp_sessions_08f6b756_758b_4f1f_bfaf_b9b5479822d7(self, method, url, body, headers):
    body = self.fixtures.load('bgp_session_get.json')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])