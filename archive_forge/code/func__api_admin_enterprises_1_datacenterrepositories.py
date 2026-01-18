import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import ET, httplib
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import NodeImage, NodeLocation
from libcloud.test.compute import TestCaseMixin
from libcloud.common.abiquo import ForbiddenError, get_href
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.abiquo import AbiquoNodeDriver
def _api_admin_enterprises_1_datacenterrepositories(self, method, url, body, headers):
    if headers['Authorization'] == 'Basic Z286dHJ1bmtz':
        expected_response = self.fixtures.load('not_found_error.xml')
        return (httplib.NOT_FOUND, expected_response, {}, '')
    elif headers['Authorization'] != 'Basic c29uOmdvaGFu':
        return (httplib.OK, self.fixtures.load('ent_1_dcreps.xml'), {}, '')
    else:
        expected_response = self.fixtures.load('privilege_errors.html')
        return (httplib.FORBIDDEN, expected_response, {}, '')