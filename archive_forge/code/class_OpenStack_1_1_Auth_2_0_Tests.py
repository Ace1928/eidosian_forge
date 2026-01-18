import os
import sys
import datetime
import unittest
from unittest import mock
from unittest.mock import Mock, patch
import pytest
import requests_mock
from libcloud.test import XML_HEADERS, MockHttp
from libcloud.pricing import set_pricing, clear_pricing_data
from libcloud.utils.py3 import u, httplib, method_type
from libcloud.common.base import LibcloudConnection
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.compute.base import Node, NodeSize, NodeImage
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import OPENSTACK_PARAMS
from libcloud.compute.types import (
from libcloud.utils.iso8601 import UTC
from libcloud.common.exceptions import BaseHTTPError
from libcloud.compute.providers import get_driver
from libcloud.test.file_fixtures import OpenStackFixtures, ComputeFileFixtures
from libcloud.common.openstack_identity import (
from libcloud.compute.drivers.openstack import (
class OpenStack_1_1_Auth_2_0_Tests(OpenStack_1_1_Tests):
    driver_args = OPENSTACK_PARAMS + ('1.1',)
    driver_kwargs = {'ex_force_auth_version': '2.0'}

    def setUp(self):
        self.driver_klass.connectionCls.conn_class = OpenStack_2_0_MockHttp
        self.driver_klass.connectionCls.auth_url = 'https://auth.api.example.com'
        OpenStackMockHttp.type = None
        OpenStack_1_1_MockHttp.type = None
        OpenStack_2_0_MockHttp.type = None
        self.driver = self.create_driver()
        self.driver.connection._populate_hosts_and_request_paths()
        clear_pricing_data()
        self.node = self.driver.list_nodes()[1]

    def test_auth_user_info_is_set(self):
        self.driver.connection._populate_hosts_and_request_paths()
        self.assertEqual(self.driver.connection.auth_user_info, {'id': '7', 'name': 'testuser', 'roles': [{'description': 'Default Role.', 'id': 'identity:default', 'name': 'identity:default'}]})