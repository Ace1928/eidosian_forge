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
def _objects_storages_690de890_13c0_4e76_8a01_e10ba8786e53_POST(self, method, url, body, headers):
    body = self.fixtures.load('create_volume_response_dict.json')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])