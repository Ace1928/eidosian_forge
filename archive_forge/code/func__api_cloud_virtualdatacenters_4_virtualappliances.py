import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import ET, httplib
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import NodeImage, NodeLocation
from libcloud.test.compute import TestCaseMixin
from libcloud.common.abiquo import ForbiddenError, get_href
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.abiquo import AbiquoNodeDriver
def _api_cloud_virtualdatacenters_4_virtualappliances(self, method, url, body, headers):
    if method == 'POST':
        vapp_name = ET.XML(body).findtext('name')
        if vapp_name == 'libcloud_test_group':
            response = self.fixtures.load('vdc_4_vapp_creation_ok.xml')
            return (httplib.OK, response, {}, '')
        elif vapp_name == 'new_group_name':
            response = self.fixtures.load('vdc_4_vapp_creation_ok.xml')
            return (httplib.OK, response, {}, '')
    else:
        return (httplib.OK, self.fixtures.load('vdc_4_vapps.xml'), {}, '')