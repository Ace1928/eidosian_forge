import sys
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.common.types import ServiceUnavailableError
from libcloud.compute.base import NodeSize, NodeImage
from libcloud.test.secrets import VULTR_PARAMS
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vultr import VultrNodeDriver, VultrNodeDriverV1
def _v1_sshkey_create(self, method, url, body, headers):
    body = self.fixtures.load('create_key_pair.json')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])