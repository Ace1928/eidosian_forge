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
def _metal_v1_ssh_keys_2c1a7f23_1dc6_4a37_948e_d9857d9f607c(self, method, url, body, headers):
    if method == 'DELETE':
        return (httplib.OK, '', {}, httplib.responses[httplib.OK])