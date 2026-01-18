import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_KUBERNETES
from libcloud.container.base import ContainerImage
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.test.common.test_kubernetes import KubernetesAuthTestCaseMixin
from libcloud.container.drivers.kubernetes import (
def _api_v1_nodes(self, method, url, body, headers):
    if method == 'GET':
        body = self.fixtures.load('_api_v1_nodes.json')
    else:
        raise AssertionError('Unsupported method')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])