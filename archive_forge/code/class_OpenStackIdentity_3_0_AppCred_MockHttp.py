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
class OpenStackIdentity_3_0_AppCred_MockHttp(OpenStackIdentity_3_0_MockHttp):

    def _v3_auth_tokens(self, method, url, body, headers):
        if method == 'POST':
            status = httplib.OK
            data = json.loads(body)
            if 'application_credential' not in data['auth']['identity']['methods']:
                status = httplib.UNAUTHORIZED
            else:
                appcred = data['auth']['identity']['application_credential']
                if appcred['id'] != 'appcred_id' or appcred['secret'] != 'appcred_secret':
                    status = httplib.UNAUTHORIZED
            body = ComputeFileFixtures('openstack').load('_v3__auth.json')
            headers = self.json_content_headers.copy()
            headers['x-subject-token'] = '00000000000000000000000000000000'
            return (status, body, headers, httplib.responses[httplib.OK])
        raise NotImplementedError()