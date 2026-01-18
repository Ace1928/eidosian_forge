import sys
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CLOUDSCALE_PARAMS
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudscale import CloudscaleNodeDriver
def _v1_servers_47cec963_fcd2_482f_bdb6_24461b2d47b1_reboot(self, method, url, body, headers):
    return (httplib.OK, '', {}, httplib.responses[httplib.OK])