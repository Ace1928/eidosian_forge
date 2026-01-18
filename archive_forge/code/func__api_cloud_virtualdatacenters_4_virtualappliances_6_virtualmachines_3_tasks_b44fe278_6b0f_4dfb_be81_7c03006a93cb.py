import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import ET, httplib
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import NodeImage, NodeLocation
from libcloud.test.compute import TestCaseMixin
from libcloud.common.abiquo import ForbiddenError, get_href
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.abiquo import AbiquoNodeDriver
def _api_cloud_virtualdatacenters_4_virtualappliances_6_virtualmachines_3_tasks_b44fe278_6b0f_4dfb_be81_7c03006a93cb(self, method, url, body, headers):
    if headers['Authorization'] == 'Basic dGVuOnNoaW4=':
        response = self.fixtures.load('vdc_4_vapp_6_vm_3_deploy_task_failed.xml')
    else:
        response = self.fixtures.load('vdc_4_vapp_6_vm_3_deploy_task.xml')
    return (httplib.OK, response, {}, '')