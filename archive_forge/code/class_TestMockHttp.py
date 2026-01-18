import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, Connection, XmlResponse, JsonResponse
from libcloud.test.file_fixtures import ComputeFileFixtures
class TestMockHttp(MockHttp):
    fixtures = ComputeFileFixtures('meta')

    def _unicode(self, method, url, body, headers):
        body = self.fixtures.load('unicode.txt')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _unicode_json(self, method, url, body, headers):
        body = self.fixtures.load('unicode.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _unicode_xml(self, method, url, body, headers):
        body = self.fixtures.load('unicode.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _ascii(self, method, url, body, headers):
        body = self.fixtures.load('helloworld.txt')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])