import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import ET, httplib
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import NodeImage, NodeLocation
from libcloud.test.compute import TestCaseMixin
from libcloud.common.abiquo import ForbiddenError, get_href
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.abiquo import AbiquoNodeDriver
def _api_cloud_virtualdatacenters_4_virtualappliances_6_virtualmachines_3_tasks_1da8c8b6_86f6_49ef_9d29_57dcc73b875a(self, method, url, body, headers):
    if headers['Authorization'] == 'Basic bXV0ZW46cm9zaGk=':
        response = self.fixtures.load('vdc_4_vapp_6_undeploy_task_failed.xml')
    else:
        response = self.fixtures.load('vdc_4_vapp_6_undeploy_task.xml')
    return (httplib.OK, response, {}, '')