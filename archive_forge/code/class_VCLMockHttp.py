import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib, xmlrpclib
from libcloud.test.secrets import VCL_PARAMS
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vcl import VCLNodeDriver as VCL
class VCLMockHttp(MockHttp):
    fixtures = ComputeFileFixtures('vcl')

    def _get_method_name(self, type, use_param, qs, path):
        return '_xmlrpc'

    def _xmlrpc(self, method, url, body, headers):
        params, meth_name = xmlrpclib.loads(body)
        if self.type:
            meth_name = '{}_{}'.format(meth_name, self.type)
        return getattr(self, meth_name)(method, url, body, headers)

    def XMLRPCgetImages(self, method, url, body, headers):
        body = self.fixtures.load('XMLRPCgetImages.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def XMLRPCextendRequest(self, method, url, body, headers):
        body = self.fixtures.load('XMLRPCextendRequest.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def XMLRPCgetRequestIds(self, method, url, body, headers):
        body = self.fixtures.load('XMLRPCgetRequestIds.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def XMLRPCgetRequestStatus(self, method, url, body, headers):
        body = self.fixtures.load('XMLRPCgetRequestStatus.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def XMLRPCendRequest(self, method, url, body, headers):
        body = self.fixtures.load('XMLRPCendRequest.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def XMLRPCaddRequest(self, method, url, body, headers):
        body = self.fixtures.load('XMLRPCaddRequest.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def XMLRPCgetRequestConnectData(self, method, url, body, headers):
        body = self.fixtures.load('XMLRPCgetRequestConnectData.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])