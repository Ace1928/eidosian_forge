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
def _metal_v1_ips_aea4ee0c_675f_4b77_8337_8e13b868dd9c(self, method, url, body, headers):
    if method == 'DELETE':
        return (httplib.OK, '', {}, httplib.responses[httplib.OK])