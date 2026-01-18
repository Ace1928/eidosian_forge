import sys
import json
import unittest
import libcloud.compute.drivers.equinixmetal
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, KeyPair
from libcloud.test.compute import TestCaseMixin
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.equinixmetal import EquinixMetalNodeDriver
class EquinixMetalMockHttp(MockHttp):
    fixtures = ComputeFileFixtures('equinixmetal')

    def _metal_v1_facilities(self, method, url, body, headers):
        body = self.fixtures.load('facilities.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _metal_v1_plans(self, method, url, body, headers):
        body = self.fixtures.load('plans.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _metal_v1_projects_3d27fd13_0466_4878_be22_9a4b5595a3df_plans(self, method, url, body, headers):
        body = self.fixtures.load('plans.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _metal_v1_projects(self, method, url, body, headers):
        body = self.fixtures.load('projects.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _metal_v1_projects_4b653fce_6405_4300_9f7d_c587b7888fe5_devices(self, method, url, body, headers):
        body = self.fixtures.load('devices_for_project.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _metal_v1_projects_4a4bce6b_d2ef_41f8_95cf_0e2f32996440_devices(self, method, url, body, headers):
        body = self.fixtures.load('devices_for_project.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _metal_v1_projects_3d27fd13_0466_4878_be22_9a4b5595a3df_devices(self, method, url, body, headers):
        body = self.fixtures.load('devices_for_project.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _metal_v1_projects_4b653fce_6405_4300_9f7d_c587b7888fe5_ips(self, method, url, body, headers):
        body = self.fixtures.load('project_ips.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _metal_v1_projects_3d27fd13_0466_4878_be22_9a4b5595a3df_ips(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('reserve_ip.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _metal_v1_projects_4b653fce_6405_4300_9f7d_c587b7888fe5_bgp_config(self, method, url, body, headers):
        body = self.fixtures.load('bgp_config_project_1.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _metal_v1_projects_3d27fd13_0466_4878_be22_9a4b5595a3df_bgp_config(self, method, url, body, headers):
        body = self.fixtures.load('bgp_config_project_1.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _metal_v1_projects_4a4bce6b_d2ef_41f8_95cf_0e2f32996440_bgp_config(self, method, url, body, headers):
        body = self.fixtures.load('bgp_config_project_3.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _metal_v1_operating_systems(self, method, url, body, headers):
        body = self.fixtures.load('operatingsystems.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _metal_v1_ssh_keys(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('sshkeys.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        if method == 'POST':
            body = self.fixtures.load('sshkey_create.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _metal_v1_ssh_keys_2c1a7f23_1dc6_4a37_948e_d9857d9f607c(self, method, url, body, headers):
        if method == 'DELETE':
            return (httplib.OK, '', {}, httplib.responses[httplib.OK])

    def _metal_v1_projects_project_id_devices(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('device_create.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        elif method == 'GET':
            body = self.fixtures.load('devices.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _metal_v1_devices_1e52437e_bbbb_cccc_dddd_74a9dfd3d3bb(self, method, url, body, headers):
        if method in ['DELETE', 'PUT']:
            return (httplib.OK, '', {}, httplib.responses[httplib.OK])

    def _metal_v1_devices_1e52437e_bbbb_cccc_dddd_74a9dfd3d3bb_actions(self, method, url, body, headers):
        return (httplib.OK, '', {}, httplib.responses[httplib.OK])

    def _metal_v1_devices_1e52437e_bbbb_cccc_dddd_74a9dfd3d3bb_bgp_sessions(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('bgp_session_create.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _metal_v1_bgp_sessions_08f6b756_758b_4f1f_bfaf_b9b5479822d7(self, method, url, body, headers):
        body = self.fixtures.load('bgp_session_get.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _metal_v1_projects_4b653fce_6405_4300_9f7d_c587b7888fe5_bgp_sessions(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('bgp_sessions.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _metal_v1_devices_905037a4_967c_4e81_b364_3a0603aa071b_bgp_sessions(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('bgp_sessions.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _metal_v1_projects_4a4bce6b_d2ef_41f8_95cf_0e2f32996440_bgp_sessions(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('bgp_sessions.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _metal_v1_projects_3d27fd13_0466_4878_be22_9a4b5595a3df_bgp_sessions(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('bgp_sessions.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _metal_v1_projects_3d27fd13_0466_4878_be22_9a4b5595a3df_events(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('project_events.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _metal_v1_devices_905037a4_967c_4e81_b364_3a0603aa071b_events(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('device_events.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _metal_v1_devices_1e52437e_bbbb_cccc_dddd_74a9dfd3d3bb_bandwidth(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('node_bandwidth.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _metal_v1_ips_01c184f5_1413_4b0b_9f6d_ac993f6c9241(self, method, url, body, headers):
        body = self.fixtures.load('ip_address.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _metal_v1_devices_1e52437e_bbbb_cccc_dddd_74a9dfd3d3bb_ips(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('ip_assignments.json')
        elif method == 'POST':
            body = self.fixtures.load('associate_ip.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _metal_v1_ips_aea4ee0c_675f_4b77_8337_8e13b868dd9c(self, method, url, body, headers):
        if method == 'DELETE':
            return (httplib.OK, '', {}, httplib.responses[httplib.OK])

    def _metal_v1_projects_3d27fd13_0466_4878_be22_9a4b5595a3df_storage(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('volumes.json')
        elif method == 'POST':
            body = self.fixtures.load('create_volume.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _metal_v1_projects_4a4bce6b_d2ef_41f8_95cf_0e2f32996440_storage(self, method, url, body, headers):
        if method == 'GET':
            body = json.dumps({'volumes': []})
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _metal_v1_projects_4b653fce_6405_4300_9f7d_c587b7888fe5_storage(self, method, url, body, headers):
        if method == 'GET':
            body = json.dumps({'volumes': []})
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _metal_v1_storage_74f11291_fde8_4abf_8150_e51cda7308c3(self, method, url, body, headers):
        if method == 'DELETE':
            return (httplib.NO_CONTENT, '', {}, httplib.responses[httplib.NO_CONTENT])

    def _metal_v1_storage_a08aaf76_e0ce_43aa_b9cd_cce0d4ae4f4c_attachments(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('attach_volume.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _metal_v1_storage_a08aaf76_e0ce_43aa_b9cd_cce0d4ae4f4c(self, method, url, body, headers):
        if method == 'DELETE':
            return (httplib.NO_CONTENT, '', {}, httplib.responses[httplib.NO_CONTENT])

    def _metal_v1_storage_attachments_2c16a96f_bb4f_471b_8e2e_b5820b9e1603(self, method, url, body, headers):
        if method == 'DELETE':
            return (httplib.NO_CONTENT, '', {}, httplib.responses[httplib.NO_CONTENT])