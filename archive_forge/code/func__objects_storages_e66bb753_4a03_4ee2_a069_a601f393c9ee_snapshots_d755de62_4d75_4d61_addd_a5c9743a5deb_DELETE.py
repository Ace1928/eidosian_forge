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
def _objects_storages_e66bb753_4a03_4ee2_a069_a601f393c9ee_snapshots_d755de62_4d75_4d61_addd_a5c9743a5deb_DELETE(self, method, url, body, headers):
    if method == 'DELETE':
        return (httplib.NO_CONTENT, None, {}, httplib.responses[httplib.NO_CONTENT])
    else:
        raise ValueError('Invalid method')