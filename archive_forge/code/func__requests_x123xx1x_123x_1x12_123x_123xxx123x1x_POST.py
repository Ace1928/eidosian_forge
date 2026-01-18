import sys
import unittest
from datetime import datetime
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.compute.base import NodeSize
from libcloud.test.secrets import GRIDSCALE_PARAMS
from libcloud.utils.iso8601 import UTC
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.gridscale import GridscaleNodeDriver
def _requests_x123xx1x_123x_1x12_123x_123xxx123x1x_POST(self, method, url, body, headers):
    body = self.fixtures.load('create_node_response_dict.json')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])