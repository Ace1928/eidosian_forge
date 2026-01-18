import sys
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node
from libcloud.test.secrets import ONAPP_PARAMS
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.onapp import OnAppNodeDriver
def _virtual_machines_json(self, method, url, body, headers):
    if method == 'GET':
        body = self.fixtures.load('list_nodes.json')
    else:
        body = self.fixtures.load('create_node.json')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])