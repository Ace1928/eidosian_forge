import sys
import unittest
import libcloud.compute.drivers.opennebula
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeState
from libcloud.test.secrets import OPENNEBULA_PARAMS
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.opennebula import (
def _compute_25(self, method, url, body, headers):
    """
        Compute entry resource.
        """
    if method == 'GET':
        body = self.fixtures.load('compute_25.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])
    if method == 'PUT':
        body = ''
        return (httplib.ACCEPTED, body, {}, httplib.responses[httplib.ACCEPTED])
    if method == 'DELETE':
        body = ''
        return (httplib.NO_CONTENT, body, {}, httplib.responses[httplib.NO_CONTENT])