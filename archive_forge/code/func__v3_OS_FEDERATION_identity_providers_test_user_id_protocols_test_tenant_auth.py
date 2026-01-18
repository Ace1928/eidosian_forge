import sys
import datetime
from unittest.mock import Mock
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.test.secrets import OPENSTACK_PARAMS
from libcloud.common.openstack import OpenStackBaseConnection
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.common.openstack_identity import (
from libcloud.compute.drivers.openstack import OpenStack_1_0_NodeDriver
from libcloud.test.compute.test_openstack import (
def _v3_OS_FEDERATION_identity_providers_test_user_id_protocols_test_tenant_auth(self, method, url, body, headers):
    if method == 'GET':
        if 'Authorization' not in headers:
            return (httplib.UNAUTHORIZED, '', headers, httplib.responses[httplib.OK])
        if headers['Authorization'] == 'Bearer test_key':
            response_body = ComputeFileFixtures('openstack').load('_v3__auth.json')
            response_headers = {'Content-Type': 'application/json', 'x-subject-token': 'foo-bar'}
            return (httplib.OK, response_body, response_headers, httplib.responses[httplib.OK])
        return (httplib.UNAUTHORIZED, '{}', headers, httplib.responses[httplib.OK])
    raise NotImplementedError()