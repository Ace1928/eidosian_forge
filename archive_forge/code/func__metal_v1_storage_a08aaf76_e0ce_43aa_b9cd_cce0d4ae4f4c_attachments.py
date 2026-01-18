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
def _metal_v1_storage_a08aaf76_e0ce_43aa_b9cd_cce0d4ae4f4c_attachments(self, method, url, body, headers):
    if method == 'POST':
        body = self.fixtures.load('attach_volume.json')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])