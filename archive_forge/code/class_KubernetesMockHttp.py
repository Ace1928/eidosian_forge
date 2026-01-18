import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_KUBERNETES
from libcloud.container.base import ContainerImage
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.test.common.test_kubernetes import KubernetesAuthTestCaseMixin
from libcloud.container.drivers.kubernetes import (
class KubernetesMockHttp(MockHttp):
    fixtures = ContainerFileFixtures('kubernetes')

    def _api_v1_pods(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_api_v1_pods.json')
        else:
            raise AssertionError('Unsupported method')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_v1_namespaces(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_api_v1_namespaces.json')
        elif method == 'POST':
            body = self.fixtures.load('_api_v1_namespaces_test.json')
        else:
            raise AssertionError('Unsupported method')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_v1_namespaces_default(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_api_v1_namespaces_default.json')
        elif method == 'DELETE':
            body = self.fixtures.load('_api_v1_namespaces_default_DELETE.json')
        else:
            raise AssertionError('Unsupported method')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_v1_namespaces_default_pods(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_api_v1_namespaces_default_pods.json')
        elif method == 'POST':
            body = self.fixtures.load('_api_v1_namespaces_default_pods_POST.json')
        else:
            raise AssertionError('Unsupported method')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_v1_namespaces_default_pods_default(self, method, url, body, headers):
        if method == 'DELETE':
            body = None
        else:
            raise AssertionError('Unsupported method')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_v1_nodes(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_api_v1_nodes.json')
        else:
            raise AssertionError('Unsupported method')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_v1_nodes_127_0_0_1(self, method, url, body, headers):
        if method == 'DELETE':
            body = None
        else:
            raise AssertionError('Unsupported method')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_v1_services(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_api_v1_services.json')
        else:
            raise AssertionError('Unsupported method')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _version(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_version.json')
        else:
            raise AssertionError('Unsupported method')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _apis_metrics_k8s_io_v1beta1_nodes(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_apis_metrics_k8s_io_v1beta1_nodes.json')
        else:
            raise AssertionError('Unsupported method')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _apis_metrics_k8s_io_v1beta1_pods(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_apis_metrics_k8s_io_v1beta1_pods.json')
        else:
            raise AssertionError('Unsupported method')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _apis_apps_v1_deployments(self, method, url, body, headers):
        body = self.fixtures.load('_apis_apps_v1_deployments.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])