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
def _1_2_server_00f8c525_7e62_4108_8115_3958df5b43dc(self, method, url, body, headers):
    body = self.fixtures.load('api_1_2_server_00f8c525-7e62-4108-8115-3958df5b43dc.json')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])