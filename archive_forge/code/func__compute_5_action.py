import sys
import unittest
import libcloud.compute.drivers.opennebula
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeState
from libcloud.test.secrets import OPENNEBULA_PARAMS
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.opennebula import (
def _compute_5_action(self, method, url, body, headers):
    body = self.fixtures_3_6.load('compute_5.xml')
    if method == 'POST':
        return (httplib.ACCEPTED, body, {}, httplib.responses[httplib.ACCEPTED])
    if method == 'GET':
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])