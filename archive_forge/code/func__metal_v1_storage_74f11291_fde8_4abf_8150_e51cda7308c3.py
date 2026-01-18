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
def _metal_v1_storage_74f11291_fde8_4abf_8150_e51cda7308c3(self, method, url, body, headers):
    if method == 'DELETE':
        return (httplib.NO_CONTENT, '', {}, httplib.responses[httplib.NO_CONTENT])