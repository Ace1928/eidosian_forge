import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vsphere import VSphere_REST_NodeDriver
def _rest_vcenter_vm_vm_80(self, method, url, body, headers):
    if method == 'GET':
        body = self.fixtures.load('node_80.json')
    elif method == 'POST':
        return
    elif method == 'DELETE':
        body = ''
    else:
        raise AssertionError('Unsupported method')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])