import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.common.vultr import VultrException
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vultr import VultrNodeDriver, VultrNodeDriverV2
def _v2_blocks_1234_attach_WRONG_LOCATION(self, method, url, body, headers):
    body = '{"error": "unable to attach: Block storage volume must be in the same region as the server it is attached to.", "status": 400}'
    return (httplib.BAD_REQUEST, body, {}, httplib.responses[httplib.BAD_REQUEST])