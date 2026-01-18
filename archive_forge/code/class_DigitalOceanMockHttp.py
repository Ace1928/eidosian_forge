import sys
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.common.types import InvalidCredsError
from libcloud.test.secrets import DIGITALOCEAN_v2_PARAMS
from libcloud.test.file_fixtures import FileFixtures
from libcloud.common.digitalocean import DigitalOceanBaseDriver
class DigitalOceanMockHttp(MockHttp):
    fixtures = FileFixtures('common', 'digitalocean')
    response = {None: httplib.OK, 'CREATE': httplib.CREATED, 'DELETE': httplib.NO_CONTENT, 'EMPTY': httplib.OK, 'NOT_FOUND': httplib.NOT_FOUND, 'UNAUTHORIZED': httplib.UNAUTHORIZED, 'UPDATE': httplib.OK}

    def _v2_account(self, method, url, body, headers):
        body = self.fixtures.load('_v2_account.json')
        return (self.response[self.type], body, {}, httplib.responses[self.response[self.type]])

    def _v2_account_UNAUTHORIZED(self, method, url, body, headers):
        body = self.fixtures.load('_v2_account_UNAUTHORIZED.json')
        return (self.response[self.type], body, {}, httplib.responses[self.response[self.type]])

    def _v2_actions(self, method, url, body, headers):
        body = self.fixtures.load('_v2_actions.json')
        return (self.response[self.type], body, {}, httplib.responses[self.response[self.type]])

    def _v2_actions_12345670(self, method, url, body, headers):
        body = self.fixtures.load('_v2_actions_12345670.json')
        return (self.response[self.type], body, {}, httplib.responses[self.response[self.type]])

    def _v2_actions_page_1(self, method, url, body, headers):
        body = self.fixtures.load('_v2_actions_page_1.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])