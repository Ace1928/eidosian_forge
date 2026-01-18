import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.kubevirt import KubeVirtNodeDriver
from libcloud.test.common.test_kubernetes import KubernetesAuthTestCaseMixin
class KubeVirtMockHttp(MockHttp):
    fixtures = ComputeFileFixtures('kubevirt')

    def _api_v1_namespaces(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_api_v1_namespaces.json')
        else:
            raise AssertionError('Unsupported method')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _apis_kubevirt_io_v1alpha3_namespaces_default_virtualmachines(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('get_default_vms.json')
            resp = httplib.OK
        elif method == 'POST':
            body = self.fixtures.load('create_vm.json')
            resp = httplib.CREATED
        else:
            AssertionError('Unsupported method')
        return (resp, body, {}, httplib.responses[httplib.OK])

    def _apis_kubevirt_io_v1alpha3_namespaces_kube_node_lease_virtualmachines(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('get_kube_node_lease_vms.json')
        elif method == 'POST':
            pass
        else:
            AssertionError('Unsupported method')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _apis_kubevirt_io_v1alpha3_namespaces_kube_public_virtualmachines(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('get_kube_public_vms.json')
        elif method == 'POST':
            pass
        else:
            AssertionError('Unsupported method')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _apis_kubevirt_io_v1alpha3_namespaces_kube_system_virtualmachines(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('get_kube_system_vms.json')
        elif method == 'POST':
            pass
        else:
            AssertionError('Unsupported method')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _apis_kubevirt_io_v1alpha3_namespaces_kubevirt_virtualmachines(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('get_kube_public_vms.json')
        elif method == 'POST':
            pass
        else:
            AssertionError('Unsupported method')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _apis_kubevirt_io_v1alpha3_namespaces_default_virtualmachines_testvm(self, method, url, body, headers):
        header = 'application/merge-patch+json'
        data_stop = {'spec': {'running': False}}
        data_start = {'spec': {'running': True}}
        if method == 'PATCH' and headers['Content-Type'] == header and (body == data_start):
            body = self.fixtures.load('start_testvm.json')
        elif method == 'PATCH' and headers['Content-Type'] == header and (body == data_stop):
            body = self.fixtures.load('stop_testvm.json')
        else:
            AssertionError('Unsupported method')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

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

    def _apis_kubevirt_io_v1alpha3_namespaces_default_virtualmachineinstances_testvm(self, method, url, body, headers):
        if method == 'DELETE':
            body = self.fixtures.load('delete_vmi_testvm.json')
        else:
            AssertionError('Unsupported method')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_v1_namespaces_default_pods(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('get_pods.json')
        else:
            AssertionError('Unsupported method')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_v1_namespaces_default_services(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('get_services.json')
        else:
            AssertionError('Unsupported method')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])