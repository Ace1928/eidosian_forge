import sys
import unittest
import libcloud.compute.drivers.opennebula
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeState
from libcloud.test.secrets import OPENNEBULA_PARAMS
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.opennebula import (
def _storage_15(self, method, url, body, headers):
    """
        Storage entry resource.
        """
    if method == 'GET':
        body = self.fixtures_3_6.load('disk_15.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])