import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node
from libcloud.test.compute import TestCaseMixin
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.maxihost import MaxihostNodeDriver
class MaxihostMockHttp(MockHttp):
    fixtures = ComputeFileFixtures('maxihost')

    def _plans(self, method, url, body, headers):
        body = self.fixtures.load('plans.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _regions(self, method, url, body, headers):
        body = self.fixtures.load('regions.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _plans_operating_systems(self, method, url, body, headers):
        body = self.fixtures.load('images.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _devices(self, method, url, body, headers):
        body = self.fixtures.load('nodes.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _devices_1319(self, method, url, body, headers):
        if method == 'DELETE':
            body = '{}'
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        else:
            raise ValueError('Unsupported method: %s' % method)

    def _devices_1319_actions(self, method, url, body, headers):
        body = self.fixtures.load('node.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _account_keys(self, method, url, body, headers):
        body = self.fixtures.load('keys.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])