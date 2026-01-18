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
class VCloud_1_5_MockHttp(MockHttp, unittest.TestCase):
    fixtures = ComputeFileFixtures('vcloud_1_5')

    def request(self, method, url, body=None, headers=None, raw=False, stream=False):
        self.assertTrue(url.startswith('/api/'), ('"%s" is invalid. Needs to start with "/api". The passed URL should be just the path, not full URL.', url))
        super().request(method, url, body, headers, raw)

    def _api_sessions(self, method, url, body, headers):
        headers['x-vcloud-authorization'] = 'testtoken'
        body = self.fixtures.load('api_sessions.xml')
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _api_org(self, method, url, body, headers):
        body = self.fixtures.load('api_org.xml')
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _api_org_96726c78_4ae3_402f_b08b_7a78c6903d2a(self, method, url, body, headers):
        body = self.fixtures.load('api_org_96726c78_4ae3_402f_b08b_7a78c6903d2a.xml')
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _api_network_dca8b667_6c8f_4c3e_be57_7a9425dba4f4(self, method, url, body, headers):
        body = self.fixtures.load('api_network_dca8b667_6c8f_4c3e_be57_7a9425dba4f4.xml')
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _api_vdc_3d9ae28c_1de9_4307_8107_9356ff8ba6d0(self, method, url, body, headers):
        body = self.fixtures.load('api_vdc_3d9ae28c_1de9_4307_8107_9356ff8ba6d0.xml')
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _api_vdc_brokenVdc(self, method, url, body, headers):
        body = self.fixtures.load('api_vdc_brokenVdc.xml')
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _api_vApp_vapp_errorRaiser(self, method, url, body, headers):
        m = AnotherErrorMember()
        raise AnotherError(m)

    def _api_vdc_3d9ae28c_1de9_4307_8107_9356ff8ba6d0_action_instantiateVAppTemplate(self, method, url, body, headers):
        body = self.fixtures.load('api_vdc_3d9ae28c_1de9_4307_8107_9356ff8ba6d0_action_instantiateVAppTemplate.xml')
        return (httplib.ACCEPTED, body, headers, httplib.responses[httplib.ACCEPTED])

    def _api_vApp_vapp_8c57a5b6_e61b_48ca_8a78_3b70ee65ef6a_power_action_powerOn(self, method, url, body, headers):
        return self._api_vApp_vapp_8c57a5b6_e61b_48ca_8a78_3b70ee65ef6b_power_action_all(method, url, body, headers)

    def _api_vdc_3d9ae28c_1de9_4307_8107_9356ff8ba6d0_action_cloneVApp(self, method, url, body, headers):
        body = self.fixtures.load('api_vdc_3d9ae28c_1de9_4307_8107_9356ff8ba6d0_action_cloneVApp.xml')
        return (httplib.ACCEPTED, body, headers, httplib.responses[httplib.ACCEPTED])

    def _api_vApp_vm_dd75d1d3_5b7b_48f0_aff3_69622ab7e045_networkConnectionSection(self, method, url, body, headers):
        body = self.fixtures.load('api_task_b034df55_fe81_4798_bc81_1f0fd0ead450.xml')
        return (httplib.ACCEPTED, body, headers, httplib.responses[httplib.ACCEPTED])

    def _api_vApp_vapp_8c57a5b6_e61b_48ca_8a78_3b70ee65ef6a(self, method, url, body, headers):
        status = httplib.OK
        if method == 'GET':
            body = self.fixtures.load('api_vApp_vapp_8c57a5b6_e61b_48ca_8a78_3b70ee65ef6a.xml')
            status = httplib.OK
        elif method == 'DELETE':
            body = self.fixtures.load('api_task_b034df55_fe81_4798_bc81_1f0fd0ead450.xml')
            status = httplib.ACCEPTED
        return (status, body, headers, httplib.responses[status])

    def _api_vApp_vapp_8c57a5b6_e61b_48ca_8a78_3b70ee65ef6b(self, method, url, body, headers):
        body = self.fixtures.load('api_vApp_vapp_8c57a5b6_e61b_48ca_8a78_3b70ee65ef6b.xml')
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _api_vApp_vapp_8c57a5b6_e61b_48ca_8a78_3b70ee65ef6c(self, method, url, body, headers):
        body = self.fixtures.load('api_vApp_vapp_8c57a5b6_e61b_48ca_8a78_3b70ee65ef6c.xml')
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _api_vApp_vm_dd75d1d3_5b7b_48f0_aff3_69622ab7e045(self, method, url, body, headers):
        body = self.fixtures.load('put_api_vApp_vm_dd75d1d3_5b7b_48f0_aff3_69622ab7e045_guestCustomizationSection.xml')
        return (httplib.ACCEPTED, body, headers, httplib.responses[httplib.ACCEPTED])

    def _api_vApp_vm_dd75d1d3_5b7b_48f0_aff3_69622ab7e045_guestCustomizationSection(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('get_api_vApp_vm_dd75d1d3_5b7b_48f0_aff3_69622ab7e045_guestCustomizationSection.xml')
            status = httplib.OK
        else:
            body = self.fixtures.load('put_api_vApp_vm_dd75d1d3_5b7b_48f0_aff3_69622ab7e045_guestCustomizationSection.xml')
            status = httplib.ACCEPTED
        return (status, body, headers, httplib.responses[status])

    def _api_vApp_vapp_8c57a5b6_e61b_48ca_8a78_3b70ee65ef6a_power_action_reset(self, method, url, body, headers):
        return self._api_vApp_vapp_8c57a5b6_e61b_48ca_8a78_3b70ee65ef6b_power_action_all(method, url, body, headers)

    def _api_task_b034df55_fe81_4798_bc81_1f0fd0ead450(self, method, url, body, headers):
        body = self.fixtures.load('api_task_b034df55_fe81_4798_bc81_1f0fd0ead450.xml')
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _api_catalog_cddb3cb2_3394_4b14_b831_11fbc4028da4(self, method, url, body, headers):
        body = self.fixtures.load('api_catalog_cddb3cb2_3394_4b14_b831_11fbc4028da4.xml')
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _api_catalogItem_3132e037_759b_4627_9056_ca66466fa607(self, method, url, body, headers):
        body = self.fixtures.load('api_catalogItem_3132e037_759b_4627_9056_ca66466fa607.xml')
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _api_vApp_deployTest(self, method, url, body, headers):
        body = self.fixtures.load('api_task_deploy.xml')
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _api_vApp_vapp_8c57a5b6_e61b_48ca_8a78_3b70ee65ef6a_action_deploy(self, method, url, body, headers):
        body = self.fixtures.load('api_task_deploy.xml')
        return (httplib.ACCEPTED, body, headers, httplib.responses[httplib.ACCEPTED])

    def _api_task_deploy(self, method, url, body, headers):
        body = self.fixtures.load('api_task_deploy.xml')
        return (httplib.ACCEPTED, body, headers, httplib.responses[httplib.ACCEPTED])

    def _api_vApp_undeployTest(self, method, url, body, headers):
        body = self.fixtures.load('api_vApp_undeployTest.xml')
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _api_vApp_undeployTest_action_undeploy(self, method, url, body, headers):
        body = self.fixtures.load('api_task_undeploy.xml')
        return (httplib.ACCEPTED, body, headers, httplib.responses[httplib.ACCEPTED])

    def _api_task_undeploy(self, method, url, body, headers):
        body = self.fixtures.load('api_task_undeploy.xml')
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _api_vApp_undeployErrorTest(self, method, url, body, headers):
        body = self.fixtures.load('api_vApp_undeployTest.xml')
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _api_vApp_undeployErrorTest_action_undeploy(self, method, url, body, headers):
        if b('shutdown') in b(body):
            body = self.fixtures.load('api_task_undeploy_error.xml')
        else:
            body = self.fixtures.load('api_task_undeploy.xml')
        return (httplib.ACCEPTED, body, headers, httplib.responses[httplib.ACCEPTED])

    def _api_task_undeployError(self, method, url, body, headers):
        body = self.fixtures.load('api_task_undeploy_error.xml')
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _api_vApp_undeployPowerOffTest(self, method, url, body, headers):
        return self._api_vApp_undeployTest(method, url, body, headers)

    def _api_vApp_undeployPowerOffTest_action_undeploy(self, method, url, body, headers):
        self.assertIn(b('powerOff'), b(body))
        return self._api_vApp_undeployTest_action_undeploy(method, url, body, headers)

    def _api_vApp_vapp_access_to_resource_forbidden(self, method, url, body, headers):
        raise Exception(ET.fromstring(self.fixtures.load('api_vApp_vapp_access_to_resource_forbidden.xml')))

    def _api_vApp_vm_test(self, method, url, body, headers):
        body = self.fixtures.load('api_vApp_vm_test.xml')
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _api_vApp_vm_test_virtualHardwareSection_disks(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('get_api_vApp_vm_test_virtualHardwareSection_disks.xml')
            status = httplib.OK
        else:
            body = self.fixtures.load('put_api_vApp_vm_test_virtualHardwareSection_disks.xml')
            status = httplib.ACCEPTED
        return (status, body, headers, httplib.responses[status])

    def _api_vApp_vm_test_virtualHardwareSection_cpu(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('get_api_vApp_vm_test_virtualHardwareSection_cpu.xml')
            status = httplib.OK
        else:
            body = self.fixtures.load('put_api_vApp_vm_test_virtualHardwareSection_cpu.xml')
            status = httplib.ACCEPTED
        return (status, body, headers, httplib.responses[status])

    def _api_vApp_vm_test_virtualHardwareSection_memory(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('get_api_vApp_vm_test_virtualHardwareSection_memory.xml')
            status = httplib.OK
        else:
            body = self.fixtures.load('put_api_vApp_vm_test_virtualHardwareSection_memory.xml')
            status = httplib.ACCEPTED
        return (status, body, headers, httplib.responses[status])

    def _api_vApp_vapp_8c57a5b6_e61b_48ca_8a78_3b70ee65ef6b_power_action_powerOff(self, method, url, body, headers):
        return self._api_vApp_vapp_8c57a5b6_e61b_48ca_8a78_3b70ee65ef6b_power_action_all(method, url, body, headers)

    def _api_vApp_vapp_8c57a5b6_e61b_48ca_8a78_3b70ee65ef6b_power_action_all(self, method, url, body, headers):
        assert method == 'POST'
        body = self.fixtures.load('api_vApp_vapp_8c57a5b6_e61b_48ca_8a78_3b70ee65ef6a_power_action_all.xml')
        return (httplib.ACCEPTED, body, headers, httplib.responses[httplib.ACCEPTED])

    def _api_query(self, method, url, body, headers):
        assert method == 'GET'
        if 'type=user' in url:
            self.assertTrue('page=2' in url)
            self.assertTrue('filter=(name==jrambo)' in url)
            self.assertTrue('sortDesc=startDate')
            body = self.fixtures.load('api_query_user.xml')
        elif 'type=group' in url:
            body = self.fixtures.load('api_query_group.xml')
        elif 'type=vm' in url and 'filter=(name==testVm2)' in url:
            body = self.fixtures.load('api_query_vm.xml')
        else:
            raise AssertionError('Unexpected query type')
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _api_vApp_vapp_8c57a5b6_e61b_48ca_8a78_3b70ee65ef6b_metadata(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('api_vapp_post_metadata.xml')
            return (httplib.ACCEPTED, body, headers, httplib.responses[httplib.ACCEPTED])
        else:
            body = self.fixtures.load('api_vapp_get_metadata.xml')
            return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _api_vApp_vapp_8c57a5b6_e61b_48ca_8a78_3b70ee65ef6b_controlAccess(self, method, url, body, headers):
        body = self.fixtures.load('api_vApp_vapp_8c57a5b6_e61b_48ca_8a78_3b70ee65ef6a_controlAccess.xml')
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _api_vApp_vapp_8c57a5b6_e61b_48ca_8a78_3b70ee65ef6b_action_controlAccess(self, method, url, body, headers):
        body = str(body)
        self.assertTrue(method == 'POST')
        self.assertTrue('<IsSharedToEveryone>false</IsSharedToEveryone>' in body)
        self.assertTrue('<Subject href="https://vm-vcloud/api/admin/group/b8202c48-7151-4e61-9a6c-155474c7d413" />' in body)
        self.assertTrue('<AccessLevel>FullControl</AccessLevel>' in body)
        body = self.fixtures.load('api_vApp_vapp_8c57a5b6_e61b_48ca_8a78_3b70ee65ef6a_controlAccess.xml')
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _api_admin_group_b8202c48_7151_4e61_9a6c_155474c7d413(self, method, url, body, headers):
        body = self.fixtures.load('api_admin_group_b8202c48_7151_4e61_9a6c_155474c7d413.xml')
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _api_vApp_vapp_8c57a5b6_e61b_48ca_8a78_3b70ee65ef6d(self, method, url, body, headers):
        body = self.fixtures.load('api_vApp_vapp_8c57a5b6_e61b_48ca_8a78_3b70ee65ef6d.xml')
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])