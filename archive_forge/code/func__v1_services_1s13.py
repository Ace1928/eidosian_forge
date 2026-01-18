import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_RANCHER
from libcloud.container.base import ContainerImage
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.container.drivers.rancher import RancherContainerDriver
def _v1_services_1s13(self, method, url, body, headers):
    if method == 'GET':
        return (httplib.OK, self.fixtures.load('ex_deploy_service.json'), {}, httplib.responses[httplib.OK])
    elif method == 'DELETE':
        return (httplib.OK, self.fixtures.load('ex_destroy_service.json'), {}, httplib.responses[httplib.OK])