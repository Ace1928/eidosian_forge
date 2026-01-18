import sys
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node
from libcloud.test.secrets import ONAPP_PARAMS
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.onapp import OnAppNodeDriver
class OnAppMockHttp(MockHttp):
    fixtures = ComputeFileFixtures('onapp')

    def _virtual_machines_json(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('list_nodes.json')
        else:
            body = self.fixtures.load('create_node.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _virtual_machines_identABC_json(self, method, url, body, headers):
        return (httplib.NO_CONTENT, '', {}, httplib.responses[httplib.NO_CONTENT])

    def _templates_json(self, method, url, body, headers):
        body = self.fixtures.load('list_images.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _profile_json(self, method, url, body, headers):
        body = self.fixtures.load('profile.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _users_123_ssh_keys_json(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('list_key_pairs.json')
        else:
            body = self.fixtures.load('import_key_pair.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _users_123_ssh_keys_1_json(self, method, url, body, headers):
        body = self.fixtures.load('get_key_pair.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _settings_ssh_keys_1_json(self, method, url, body, headers):
        return (httplib.NO_CONTENT, '', {}, httplib.responses[httplib.NO_CONTENT])