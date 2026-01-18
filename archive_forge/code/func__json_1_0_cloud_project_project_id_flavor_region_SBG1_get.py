import sys
import unittest
from unittest.mock import patch
from libcloud.http import LibcloudConnection
from libcloud.test import no_internet
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import OVH_PARAMS
from libcloud.common.exceptions import BaseHTTPError
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.ovh import OvhNodeDriver
from libcloud.test.common.test_ovh import BaseOvhMockHttp
def _json_1_0_cloud_project_project_id_flavor_region_SBG1_get(self, method, url, body, headers):
    body = self.fixtures.load('flavor_get.json')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])