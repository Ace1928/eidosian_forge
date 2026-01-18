import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.common.vultr import VultrException
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vultr import VultrNodeDriver, VultrNodeDriverV2
def _v2_os_PAGINATED(self, method, url, body, headers):
    if 'cursor' not in url:
        body = self.fixtures.load('list_images_paginated_1.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])
    elif 'cursor=bmV4dF9fMjMw' in url:
        body = self.fixtures.load('list_images_paginated_2.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])
    else:
        body = self.fixtures.load('list_images_paginated_3.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])