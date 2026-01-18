import sys
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CLOUDSCALE_PARAMS
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudscale import CloudscaleNodeDriver
def _v1_images(self, method, url, body, headers):
    body = self.fixtures.load('list_images.json')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])