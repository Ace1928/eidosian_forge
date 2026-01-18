import sys
import unittest
import libcloud.compute.drivers.opennebula
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeState
from libcloud.test.secrets import OPENNEBULA_PARAMS
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.opennebula import (
class OpenNebula_3_8_MockHttp(OpenNebula_3_2_MockHttp):
    """
    Mock HTTP server for testing v3.8 of the OpenNebula.org compute driver.
    """
    fixtures_3_8 = ComputeFileFixtures('opennebula_3_8')

    def _instance_type(self, method, url, body, headers):
        """
        Instance type pool.
        """
        if method == 'GET':
            body = self.fixtures_3_8.load('instance_type_collection.xml')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _instance_type_small(self, method, url, body, headers):
        """
        Small instance type.
        """
        if method == 'GET':
            body = self.fixtures_3_8.load('instance_type_small.xml')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _instance_type_medium(self, method, url, body, headers):
        """
        Medium instance type pool.
        """
        if method == 'GET':
            body = self.fixtures_3_8.load('instance_type_medium.xml')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _instance_type_large(self, method, url, body, headers):
        """
        Large instance type pool.
        """
        if method == 'GET':
            body = self.fixtures_3_8.load('instance_type_large.xml')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])