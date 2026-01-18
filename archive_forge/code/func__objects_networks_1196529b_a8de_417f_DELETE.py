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
def _objects_networks_1196529b_a8de_417f_DELETE(self, method, url, body, headers):
    if method == 'DELETE':
        return (httplib.NO_CONTENT, None, {}, httplib.responses[httplib.NO_CONTENT])
    else:
        raise ValueError('Invalid method')