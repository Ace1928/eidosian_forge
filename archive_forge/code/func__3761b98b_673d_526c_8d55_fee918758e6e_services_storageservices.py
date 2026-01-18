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
def _3761b98b_673d_526c_8d55_fee918758e6e_services_storageservices(self, method, url, body, headers):
    if '<ServiceName>testdss123</ServiceName>' in body:
        return (httplib.ACCEPTED, body, headers, httplib.responses[httplib.ACCEPTED])
    elif '<ServiceName>dss123</ServiceName>' in body:
        return (httplib.CONFLICT, body, headers, httplib.responses[httplib.CONFLICT])