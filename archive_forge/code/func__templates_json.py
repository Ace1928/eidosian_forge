import sys
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node
from libcloud.test.secrets import ONAPP_PARAMS
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.onapp import OnAppNodeDriver
def _templates_json(self, method, url, body, headers):
    body = self.fixtures.load('list_images.json')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])