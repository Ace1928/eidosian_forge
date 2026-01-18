import sys
import unittest
import libcloud.compute.drivers.opennebula
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeState
from libcloud.test.secrets import OPENNEBULA_PARAMS
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.opennebula import (
class OpenNebula_2_0_MockHttp(MockHttp):
    """
    Mock HTTP server for testing v2.0 through v3.2 of the OpenNebula.org
    compute driver.
    """
    fixtures = ComputeFileFixtures('opennebula_2_0')

    def _compute(self, method, url, body, headers):
        """
        Compute pool resources.
        """
        if method == 'GET':
            body = self.fixtures.load('compute_collection.xml')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        if method == 'POST':
            body = self.fixtures.load('compute_5.xml')
            return (httplib.CREATED, body, {}, httplib.responses[httplib.CREATED])

    def _storage(self, method, url, body, headers):
        """
        Storage pool resources.
        """
        if method == 'GET':
            body = self.fixtures.load('storage_collection.xml')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        if method == 'POST':
            body = self.fixtures.load('storage_5.xml')
            return (httplib.CREATED, body, {}, httplib.responses[httplib.CREATED])

    def _network(self, method, url, body, headers):
        """
        Network pool resources.
        """
        if method == 'GET':
            body = self.fixtures.load('network_collection.xml')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        if method == 'POST':
            body = self.fixtures.load('network_5.xml')
            return (httplib.CREATED, body, {}, httplib.responses[httplib.CREATED])

    def _compute_5(self, method, url, body, headers):
        """
        Compute entry resource.
        """
        if method == 'GET':
            body = self.fixtures.load('compute_5.xml')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        if method == 'PUT':
            body = ''
            return (httplib.ACCEPTED, body, {}, httplib.responses[httplib.ACCEPTED])
        if method == 'DELETE':
            body = ''
            return (httplib.NO_CONTENT, body, {}, httplib.responses[httplib.NO_CONTENT])

    def _compute_15(self, method, url, body, headers):
        """
        Compute entry resource.
        """
        if method == 'GET':
            body = self.fixtures.load('compute_15.xml')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        if method == 'PUT':
            body = ''
            return (httplib.ACCEPTED, body, {}, httplib.responses[httplib.ACCEPTED])
        if method == 'DELETE':
            body = ''
            return (httplib.NO_CONTENT, body, {}, httplib.responses[httplib.NO_CONTENT])

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

    def _storage_5(self, method, url, body, headers):
        """
        Storage entry resource.
        """
        if method == 'GET':
            body = self.fixtures.load('storage_5.xml')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        if method == 'DELETE':
            body = ''
            return (httplib.NO_CONTENT, body, {}, httplib.responses[httplib.NO_CONTENT])

    def _storage_15(self, method, url, body, headers):
        """
        Storage entry resource.
        """
        if method == 'GET':
            body = self.fixtures.load('storage_15.xml')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        if method == 'DELETE':
            body = ''
            return (httplib.NO_CONTENT, body, {}, httplib.responses[httplib.NO_CONTENT])

    def _network_5(self, method, url, body, headers):
        """
        Network entry resource.
        """
        if method == 'GET':
            body = self.fixtures.load('network_5.xml')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        if method == 'DELETE':
            body = ''
            return (httplib.NO_CONTENT, body, {}, httplib.responses[httplib.NO_CONTENT])

    def _network_15(self, method, url, body, headers):
        """
        Network entry resource.
        """
        if method == 'GET':
            body = self.fixtures.load('network_15.xml')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        if method == 'DELETE':
            body = ''
            return (httplib.NO_CONTENT, body, {}, httplib.responses[httplib.NO_CONTENT])