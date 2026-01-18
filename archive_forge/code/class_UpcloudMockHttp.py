import re
import sys
import json
import base64
from libcloud.test import MockHttp, LibcloudTestCase, unittest
from libcloud.compute import providers
from libcloud.utils.py3 import httplib, ensure_string
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeLocation, NodeAuthSSHKey
from libcloud.test.secrets import UPCLOUD_PARAMS
from libcloud.compute.types import Provider, NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.upcloud import UpcloudDriver, UpcloudResponse
class UpcloudMockHttp(MockHttp):
    fixtures = ComputeFileFixtures('upcloud')

    def _1_2_zone(self, method, url, body, headers):
        auth = headers['Authorization'].split(' ')[1]
        username, password = ensure_string(base64.b64decode(auth)).split(':')
        if username == 'nosuchuser' and password == 'nopwd':
            body = self.fixtures.load('api_1_2_zone_failed_auth.json')
            status_code = httplib.UNAUTHORIZED
        else:
            body = self.fixtures.load('api_1_2_zone.json')
            status_code = httplib.OK
        return (status_code, body, {}, httplib.responses[httplib.OK])

    def _1_2_plan(self, method, url, body, headers):
        body = self.fixtures.load('api_1_2_plan.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _1_2_storage_cdrom(self, method, url, body, headers):
        body = self.fixtures.load('api_1_2_storage_cdrom.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _1_2_storage_template(self, method, url, body, headers):
        body = self.fixtures.load('api_1_2_storage_template.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _1_2_price(self, method, url, body, headers):
        body = self.fixtures.load('api_1_2_price.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _1_2_server(self, method, url, body, headers):
        if method == 'POST':
            dbody = json.loads(body)
            storages = dbody['server']['storage_devices']['storage_device']
            if any(['type' in storage and storage['type'] == 'cdrom' for storage in storages]):
                body = self.fixtures.load('api_1_2_server_from_cdrom.json')
            else:
                body = self.fixtures.load('api_1_2_server_from_template.json')
        else:
            body = self.fixtures.load('api_1_2_server.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _1_2_server_00f8c525_7e62_4108_8115_3958df5b43dc(self, method, url, body, headers):
        body = self.fixtures.load('api_1_2_server_00f8c525-7e62-4108-8115-3958df5b43dc.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _1_2_server_00f8c525_7e62_4108_8115_3958df5b43dc_restart(self, method, url, body, headers):
        body = self.fixtures.load('api_1_2_server_00f8c525-7e62-4108-8115-3958df5b43dc_restart.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _1_2_server_00893c98_5d5a_4363_b177_88df518a2b60(self, method, url, body, headers):
        body = self.fixtures.load('api_1_2_server_00893c98-5d5a-4363-b177-88df518a2b60.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])