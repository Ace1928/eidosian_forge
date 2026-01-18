import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_RANCHER
from libcloud.container.base import ContainerImage
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.container.drivers.rancher import RancherContainerDriver
def _v1_containers(self, method, url, body, headers):
    if '?state=running' in url:
        return (httplib.OK, self.fixtures.load('ex_search_containers.json'), {}, httplib.responses[httplib.OK])
    elif method == 'POST':
        return (httplib.OK, self.fixtures.load('deploy_container.json'), {}, httplib.responses[httplib.OK])
    return (httplib.OK, self.fixtures.load('list_containers.json'), {}, httplib.responses[httplib.OK])