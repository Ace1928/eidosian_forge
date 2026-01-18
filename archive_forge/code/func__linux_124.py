import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_LXD
from libcloud.container.base import Container, ContainerImage
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.container.drivers.lxd import (
def _linux_124(self, method, url, body, headers):
    return (httplib.OK, self.fixtures.load('linux_124/version.json'), {}, httplib.responses[httplib.OK])