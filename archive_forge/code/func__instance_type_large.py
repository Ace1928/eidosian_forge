import sys
import unittest
import libcloud.compute.drivers.opennebula
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeState
from libcloud.test.secrets import OPENNEBULA_PARAMS
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.opennebula import (
def _instance_type_large(self, method, url, body, headers):
    """
        Large instance type pool.
        """
    if method == 'GET':
        body = self.fixtures_3_8.load('instance_type_large.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])