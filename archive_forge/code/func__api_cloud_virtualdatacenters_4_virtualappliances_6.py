import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import ET, httplib
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import NodeImage, NodeLocation
from libcloud.test.compute import TestCaseMixin
from libcloud.common.abiquo import ForbiddenError, get_href
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.abiquo import AbiquoNodeDriver
def _api_cloud_virtualdatacenters_4_virtualappliances_6(self, method, url, body, headers):
    if method == 'GET':
        response = self.fixtures.load('vdc_4_vapp_6.xml')
        return (httplib.OK, response, {}, '')
    else:
        return (httplib.NO_CONTENT, '', {}, '')