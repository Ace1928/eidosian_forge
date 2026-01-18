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
def _json_1_0_cloud_project_project_id_volume_snapshot_foo_snap_delete(self, method, url, body, headers):
    return (httplib.OK, None, {}, httplib.responses[httplib.OK])