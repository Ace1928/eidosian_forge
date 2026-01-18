import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_RANCHER
from libcloud.container.base import ContainerImage
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.container.drivers.rancher import RancherContainerDriver
def _v1_services(self, method, url, body, headers):
    if '?healthState=healthy' in url:
        return (httplib.OK, self.fixtures.load('ex_search_services.json'), {}, httplib.responses[httplib.OK])
    elif method == 'GET':
        return (httplib.OK, self.fixtures.load('ex_list_services.json'), {}, httplib.responses[httplib.OK])
    else:
        return (httplib.OK, self.fixtures.load('ex_deploy_service.json'), {}, httplib.responses[httplib.OK])