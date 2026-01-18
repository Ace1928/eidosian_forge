import re
import sys
import datetime
import unittest
import traceback
from unittest.mock import patch, mock_open
from libcloud.test import MockHttp
from libcloud.utils.py3 import ET, PY2, b, httplib, assertRaisesRegex
from libcloud.compute.base import Node, NodeImage
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import VCLOUD_PARAMS
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import UTC
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vcloud import (
class TerremarkMockHttp(MockHttp):
    fixtures = ComputeFileFixtures('terremark')

    def _api_v0_8_login(self, method, url, body, headers):
        headers['set-cookie'] = 'vcloud-token=testtoken'
        body = self.fixtures.load('api_v0_8_login.xml')
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _api_v0_8_org_240(self, method, url, body, headers):
        body = self.fixtures.load('api_v0_8_org_240.xml')
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _api_v0_8_vdc_224(self, method, url, body, headers):
        body = self.fixtures.load('api_v0_8_vdc_224.xml')
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _api_v0_8_vdc_224_catalog(self, method, url, body, headers):
        body = self.fixtures.load('api_v0_8_vdc_224_catalog.xml')
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _api_v0_8_catalogItem_5(self, method, url, body, headers):
        body = self.fixtures.load('api_v0_8_catalogItem_5.xml')
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _api_v0_8_vdc_224_action_instantiateVAppTemplate(self, method, url, body, headers):
        body = self.fixtures.load('api_v0_8_vdc_224_action_instantiateVAppTemplate.xml')
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _api_v0_8_vapp_14031_action_deploy(self, method, url, body, headers):
        body = self.fixtures.load('api_v0_8_vapp_14031_action_deploy.xml')
        return (httplib.ACCEPTED, body, headers, httplib.responses[httplib.ACCEPTED])

    def _api_v0_8_task_10496(self, method, url, body, headers):
        body = self.fixtures.load('api_v0_8_task_10496.xml')
        return (httplib.ACCEPTED, body, headers, httplib.responses[httplib.ACCEPTED])

    def _api_v0_8_vapp_14031_power_action_powerOn(self, method, url, body, headers):
        body = self.fixtures.load('api_v0_8_vapp_14031_power_action_powerOn.xml')
        return (httplib.ACCEPTED, body, headers, httplib.responses[httplib.ACCEPTED])

    def _api_v0_8_vapp_14031(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('api_v0_8_vapp_14031_get.xml')
        elif method == 'DELETE':
            body = ''
        return (httplib.ACCEPTED, body, headers, httplib.responses[httplib.ACCEPTED])

    def _api_v0_8_vapp_14031_power_action_reset(self, method, url, body, headers):
        body = self.fixtures.load('api_v0_8_vapp_14031_power_action_reset.xml')
        return (httplib.ACCEPTED, body, headers, httplib.responses[httplib.ACCEPTED])

    def _api_v0_8_vapp_14031_power_action_poweroff(self, method, url, body, headers):
        body = self.fixtures.load('api_v0_8_vapp_14031_power_action_poweroff.xml')
        return (httplib.ACCEPTED, body, headers, httplib.responses[httplib.ACCEPTED])

    def _api_v0_8_task_11001(self, method, url, body, headers):
        body = self.fixtures.load('api_v0_8_task_11001.xml')
        return (httplib.ACCEPTED, body, headers, httplib.responses[httplib.ACCEPTED])