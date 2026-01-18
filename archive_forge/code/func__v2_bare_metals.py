import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.common.vultr import VultrException
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vultr import VultrNodeDriver, VultrNodeDriverV2
def _v2_bare_metals(self, method, url, body, headers):
    if method == 'GET':
        body = self.fixtures.load('ex_list_bare_metal_nodes.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])
    elif method == 'POST':
        body = self.fixtures.load('create_node_bare_metal.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])