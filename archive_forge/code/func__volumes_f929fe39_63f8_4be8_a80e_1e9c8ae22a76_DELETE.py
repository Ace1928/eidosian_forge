import sys
import unittest
from datetime import datetime
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.compute.base import NodeImage
from libcloud.test.secrets import SCALEWAY_PARAMS
from libcloud.utils.iso8601 import UTC
from libcloud.common.exceptions import BaseHTTPError
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.scaleway import ScalewayNodeDriver
def _volumes_f929fe39_63f8_4be8_a80e_1e9c8ae22a76_DELETE(self, method, url, body, headers):
    return (httplib.NO_CONTENT, None, {}, httplib.responses[httplib.NO_CONTENT])