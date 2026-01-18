import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vsphere import VSphere_REST_NodeDriver
def _rest_vcenter_host(self, method, url, body, headers):
    if method == 'GET':
        body = self.fixtures.load('list_hosts.json')
    else:
        raise AssertionError('Unsupported method')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])