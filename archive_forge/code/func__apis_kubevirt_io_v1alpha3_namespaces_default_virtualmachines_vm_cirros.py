import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.kubevirt import KubeVirtNodeDriver
from libcloud.test.common.test_kubernetes import KubernetesAuthTestCaseMixin
def _apis_kubevirt_io_v1alpha3_namespaces_default_virtualmachines_vm_cirros(self, method, url, body, headers):
    header = 'application/merge-patch+json'
    data_stop = {'spec': {'running': False}}
    data_start = {'spec': {'running': True}}
    if method == 'PATCH' and headers['Content-Type'] == header and (body == data_start):
        body = self.fixtures.load('start_vm_cirros.json')
    elif method == 'PATCH' and headers['Content-Type'] == header and (body == data_stop):
        body = self.fixtures.load('stop_vm_cirros.json')
    else:
        AssertionError('Unsupported method')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])