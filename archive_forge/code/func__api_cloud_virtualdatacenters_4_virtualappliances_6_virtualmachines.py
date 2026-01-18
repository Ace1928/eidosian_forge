import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import ET, httplib
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import NodeImage, NodeLocation
from libcloud.test.compute import TestCaseMixin
from libcloud.common.abiquo import ForbiddenError, get_href
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.abiquo import AbiquoNodeDriver
def _api_cloud_virtualdatacenters_4_virtualappliances_6_virtualmachines(self, method, url, body, headers):
    if method == 'GET':
        if headers['Authorization'] == 'Basic dmU6Z2V0YQ==':
            response = self.fixtures.load('vdc_4_vapp_6_vms_allocated.xml')
        else:
            response = self.fixtures.load('vdc_4_vapp_6_vms.xml')
        return (httplib.OK, response, {}, '')
    else:
        response = self.fixtures.load('vdc_4_vapp_6_vm_creation_ok.xml')
        return (httplib.CREATED, response, {}, '')