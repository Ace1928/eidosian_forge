import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib, xmlrpclib
from libcloud.test.secrets import VCL_PARAMS
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vcl import VCLNodeDriver as VCL
def XMLRPCgetRequestStatus(self, method, url, body, headers):
    body = self.fixtures.load('XMLRPCgetRequestStatus.xml')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])