import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_LXD
from libcloud.container.base import Container, ContainerImage
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.container.drivers.lxd import (
def _linux_124_storage_pools_pool1(self, method, url, body, header):
    if method == 'GET':
        json = self.fixtures.load('linux_124/storage_pool_1.json')
        return (httplib.OK, json, {}, httplib.responses[httplib.OK])
    elif method == 'DELETE':
        return (httplib.OK, self.fixtures.load('linux_124/storage_pool_delete_sucess.json'), {}, httplib.responses[httplib.OK])