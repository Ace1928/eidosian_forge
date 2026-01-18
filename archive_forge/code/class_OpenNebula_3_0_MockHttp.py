import sys
import unittest
import libcloud.compute.drivers.opennebula
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeState
from libcloud.test.secrets import OPENNEBULA_PARAMS
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.opennebula import (
class OpenNebula_3_0_MockHttp(OpenNebula_2_0_MockHttp):
    """
    Mock HTTP server for testing v3.0 of the OpenNebula.org compute driver.
    """
    fixtures_3_0 = ComputeFileFixtures('opennebula_3_0')

    def _network(self, method, url, body, headers):
        """
        Network pool resources.
        """
        if method == 'GET':
            body = self.fixtures_3_0.load('network_collection.xml')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        if method == 'POST':
            body = self.fixtures.load('network_5.xml')
            return (httplib.CREATED, body, {}, httplib.responses[httplib.CREATED])

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

    def _network_15(self, method, url, body, headers):
        """
        Network entry resource.
        """
        if method == 'GET':
            body = self.fixtures_3_0.load('network_15.xml')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        if method == 'DELETE':
            body = ''
            return (httplib.NO_CONTENT, body, {}, httplib.responses[httplib.NO_CONTENT])