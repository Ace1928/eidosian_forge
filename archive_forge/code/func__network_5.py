import sys
import unittest
import libcloud.compute.drivers.opennebula
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeState
from libcloud.test.secrets import OPENNEBULA_PARAMS
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.opennebula import (
def _network_5(self, method, url, body, headers):
    """
        Network entry resource.
        """
    if method == 'GET':
        body = self.fixtures_3_0.load('network_5.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])
    if method == 'DELETE':
        body = ''
        return (httplib.NO_CONTENT, body, {}, httplib.responses[httplib.NO_CONTENT])