import os
import sys
import libcloud.security
from libcloud.test import MockHttp, LibcloudTestCase, unittest
from libcloud.utils.py3 import httplib
from libcloud.common.types import LibcloudError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeState, NodeAuthPassword
from libcloud.compute.types import Provider
from libcloud.compute.providers import get_driver
from libcloud.test.file_fixtures import ComputeFileFixtures
def _3761b98b_673d_526c_8d55_fee918758e6e_services_vmimages(self, method, url, body, headers):
    if method == 'GET':
        body = self.fixtures.load('_3761b98b_673d_526c_8d55_fee918758e6e_services_vmimages.xml')
    return (httplib.OK, body, headers, httplib.responses[httplib.OK])