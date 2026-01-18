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
class OpenStackIdentity_2_0_Connection_VOMSTests(unittest.TestCase):

    def setUp(self):
        mock_cls = OpenStackIdentity_2_0_Connection_VOMSMockHttp
        mock_cls.type = None
        OpenStackIdentity_2_0_Connection_VOMS.conn_class = mock_cls
        self.auth_instance = OpenStackIdentity_2_0_Connection_VOMS(auth_url='http://none', user_id=None, key='/tmp/proxy.pem', tenant_name='VO')
        self.auth_instance.auth_token = 'mock'

    def test_authenticate(self):
        auth = OpenStackIdentity_2_0_Connection_VOMS(auth_url='http://none', user_id=None, key='/tmp/proxy.pem', token_scope='test', tenant_name='VO')
        auth.authenticate()