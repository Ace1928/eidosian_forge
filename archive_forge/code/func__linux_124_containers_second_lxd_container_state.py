import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_LXD
from libcloud.container.base import Container, ContainerImage
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.container.drivers.lxd import (
def _linux_124_containers_second_lxd_container_state(self, method, url, body, headers):
    if method == 'PUT' or method == 'DELETE':
        json = self.fixtures.load('linux_124/background_op.json')
        return (httplib.OK, json, {}, httplib.responses[httplib.OK])
    elif method == 'GET':
        json = self.fixtures.load('linux_124/second_lxd_container.json')
        return (httplib.OK, json, {}, httplib.responses[httplib.OK])