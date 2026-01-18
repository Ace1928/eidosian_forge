import sys
import base64
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import b, httplib
from libcloud.common.types import InvalidCredsError
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import BRIGHTBOX_PARAMS
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.brightbox import BrightboxNodeDriver
def _1_0_servers(self, method, url, body, headers):
    if method == 'GET':
        return self.test_response(httplib.OK, self.fixtures.load('list_servers.json'))
    elif method == 'POST':
        body = json.loads(body)
        encoded = base64.b64encode(b(USER_DATA)).decode('ascii')
        if 'user_data' in body and body['user_data'] != encoded:
            data = '{"error_name":"dodgy user data", "errors": ["User data not encoded properly"]}'
            return self.test_response(httplib.BAD_REQUEST, data)
        if body.get('zone', '') == 'zon-remk1':
            node = json.loads(self.fixtures.load('create_server_gb1_b.json'))
        else:
            node = json.loads(self.fixtures.load('create_server_gb1_a.json'))
        node['name'] = body['name']
        if 'server_groups' in body:
            node['server_groups'] = [{'id': x} for x in body['server_groups']]
        if 'user_data' in body:
            node['user_data'] = body['user_data']
        return self.test_response(httplib.ACCEPTED, json.dumps(node))