import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vsphere import VSphere_REST_NodeDriver
def _rest_com_vmware_cis_session(self, method, url, body, headers):
    if method == 'POST':
        body = self.fixtures.load('session_token.json')
    else:
        raise AssertionError('Unsupported method')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])