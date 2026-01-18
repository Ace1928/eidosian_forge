import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.common.vultr import VultrException
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vultr import VultrNodeDriver, VultrNodeDriverV2
def _v2_ssh_keys(self, method, url, body, headers):
    if method == 'GET':
        body = self.fixtures.load('list_key_pairs.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])
    elif method == 'POST':
        body = self.fixtures.load('import_key_pair_from_string.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])