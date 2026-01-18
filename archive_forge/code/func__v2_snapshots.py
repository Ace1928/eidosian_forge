import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.common.vultr import VultrException
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vultr import VultrNodeDriver, VultrNodeDriverV2
def _v2_snapshots(self, method, url, body, headers):
    if method == 'GET':
        body = self.fixtures.load('ex_list_snapshots.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])
    elif method == 'POST':
        body = self.fixtures.load('ex_create_snapshot.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])