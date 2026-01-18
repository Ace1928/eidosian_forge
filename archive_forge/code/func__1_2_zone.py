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