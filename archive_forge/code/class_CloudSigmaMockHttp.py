import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import (
class CloudSigmaMockHttp(MockHttp, unittest.TestCase):
    fixtures = ComputeFileFixtures('cloudsigma_2_0')

    def _api_2_0_servers_detail_INVALID_CREDS(self, method, url, body, headers):
        body = self.fixtures.load('libdrives.json')
        return (httplib.UNAUTHORIZED, body, {}, httplib.responses[httplib.UNAUTHORIZED])

    def _api_2_0_libdrives(self, method, url, body, headers):
        body = self.fixtures.load('libdrives.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_2_0_servers_detail(self, method, url, body, headers):
        body = self.fixtures.load('servers_detail_mixed_state.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_2_0_servers_9de75ed6_fd33_45e2_963f_d405f31fd911(self, method, url, body, headers):
        body = ''
        return (httplib.NO_CONTENT, body, {}, httplib.responses[httplib.NO_CONTENT])

    def _api_2_0_servers(self, method, url, body, headers):
        if method == 'POST':
            parsed = json.loads(body)
            if 'vlan' in parsed['name']:
                self.assertEqual(len(parsed['nics']), 2)
                body = self.fixtures.load('servers_create_with_vlan.json')
            else:
                body = self.fixtures.load('servers_create.json')
            return (httplib.CREATED, body, {}, httplib.responses[httplib.CREATED])

    def _api_2_0_servers_9de75ed6_fd33_45e2_963f_d405f31fd911_action_start(self, method, url, body, headers):
        body = self.fixtures.load('start_success.json')
        return (httplib.ACCEPTED, body, {}, httplib.responses[httplib.ACCEPTED])

    def _api_2_0_servers_9de75ed6_fd33_45e2_963f_d405f31fd911_action_AVOID_MODE_start(self, method, url, body, headers):
        self.assertUrlContainsQueryParams(url, {'avoid': '1,2'})
        body = self.fixtures.load('start_success.json')
        return (httplib.ACCEPTED, body, {}, httplib.responses[httplib.ACCEPTED])

    def _api_2_0_servers_9de75ed6_fd33_45e2_963f_d405f31fd911_action_ALREADY_STARTED_start(self, method, url, body, headers):
        body = self.fixtures.load('start_already_started.json')
        return (httplib.FORBIDDEN, body, {}, httplib.responses[httplib.FORBIDDEN])

    def _api_2_0_servers_9de75ed6_fd33_45e2_963f_d405f31fd911_action_stop(self, method, url, body, headers):
        body = self.fixtures.load('stop_success.json')
        return (httplib.ACCEPTED, body, {}, httplib.responses[httplib.ACCEPTED])

    def _api_2_0_servers_9de75ed6_fd33_45e2_963f_d405f31fd911_action_ALREADY_STOPPED_stop(self, method, url, body, headers):
        body = self.fixtures.load('stop_already_stopped.json')
        return (httplib.FORBIDDEN, body, {}, httplib.responses[httplib.FORBIDDEN])

    def _api_2_0_servers_9de75ed6_fd33_45e2_963f_d405f31fd911_action_clone(self, method, url, body, headers):
        body = self.fixtures.load('servers_clone.json')
        return (httplib.ACCEPTED, body, {}, httplib.responses[httplib.ACCEPTED])

    def _api_2_0_servers_9de75ed6_fd33_45e2_963f_d405f31fd911_action_open_vnc(self, method, url, body, headers):
        body = self.fixtures.load('servers_open_vnc.json')
        return (httplib.ACCEPTED, body, {}, httplib.responses[httplib.ACCEPTED])

    def _api_2_0_servers_9de75ed6_fd33_45e2_963f_d405f31fd911_action_close_vnc(self, method, url, body, headers):
        body = self.fixtures.load('servers_close_vnc.json')
        return (httplib.ACCEPTED, body, {}, httplib.responses[httplib.ACCEPTED])

    def _api_2_0_servers_3df825cb_9c1b_470d_acbd_03e1a966c046(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('servers_get.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        elif method == 'PUT':
            body = ''
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_2_0_drives_detail(self, method, url, body, headers):
        body = self.fixtures.load('drives_detail.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_2_0_drives_b02311e2_a83c_4c12_af10_b30d51c86913(self, method, url, body, headers):
        body = self.fixtures.load('drives_get.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_2_0_drives_9d1d2cf3_08c1_462f_8485_f4b073560809(self, method, url, body, headers):
        body = self.fixtures.load('drives_get.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_2_0_drives_CREATE(self, method, url, body, headers):
        body = self.fixtures.load('drives_create.json')
        return (httplib.CREATED, body, {}, httplib.responses[httplib.CREATED])

    def _api_2_0_drives_9d1d2cf3_08c1_462f_8485_f4b073560809_action_clone(self, method, url, body, headers):
        body = self.fixtures.load('drives_clone.json')
        return (httplib.ACCEPTED, body, {}, httplib.responses[httplib.ACCEPTED])

    def _api_2_0_drives_5236b9ee_f735_42fd_a236_17558f9e12d3_action_clone(self, method, url, body, headers):
        body = self.fixtures.load('drives_clone.json')
        return (httplib.ACCEPTED, body, {}, httplib.responses[httplib.ACCEPTED])

    def _api_2_0_drives_b02311e2_a83c_4c12_af10_b30d51c86913_action_resize(self, method, url, body, headers):
        body = self.fixtures.load('drives_resize.json')
        return (httplib.ACCEPTED, body, {}, httplib.responses[httplib.ACCEPTED])

    def _api_2_0_drives_9d1d2cf3_08c1_462f_8485_f4b073560809_action_resize(self, method, url, body, headers):
        body = self.fixtures.load('drives_resize.json')
        return (httplib.ACCEPTED, body, {}, httplib.responses[httplib.ACCEPTED])

    def _api_2_0_fwpolicies_detail(self, method, url, body, headers):
        body = self.fixtures.load('fwpolicies_detail.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_2_0_fwpolicies_CREATE_NO_RULES(self, method, url, body, headers):
        body = self.fixtures.load('fwpolicies_create_no_rules.json')
        return (httplib.CREATED, body, {}, httplib.responses[httplib.CREATED])

    def _api_2_0_fwpolicies_CREATE_WITH_RULES(self, method, url, body, headers):
        body = self.fixtures.load('fwpolicies_create_with_rules.json')
        return (httplib.CREATED, body, {}, httplib.responses[httplib.CREATED])

    def _api_2_0_servers_9de75ed6_fd33_45e2_963f_d405f31fd911_ATTACH_POLICY(self, method, url, body, headers):
        body = self.fixtures.load('servers_attach_policy.json')
        return (httplib.CREATED, body, {}, httplib.responses[httplib.CREATED])

    def _api_2_0_fwpolicies_0e339282_0cb5_41ac_a9db_727fb62ff2dc(self, method, url, body, headers):
        if method == 'DELETE':
            body = ''
            return (httplib.NO_CONTENT, body, {}, httplib.responses[httplib.NO_CONTENT])

    def _api_2_0_tags_detail(self, method, url, body, headers):
        body = self.fixtures.load('tags_detail.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_2_0_tags(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('tags_create.json')
            return (httplib.CREATED, body, {}, httplib.responses[httplib.CREATED])

    def _api_2_0_tags_WITH_RESOURCES(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('tags_create_with_resources.json')
            return (httplib.CREATED, body, {}, httplib.responses[httplib.CREATED])

    def _api_2_0_tags_a010ec41_2ead_4630_a1d0_237fa77e4d4d(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('tags_get.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        elif method == 'PUT':
            body = self.fixtures.load('tags_update.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        elif method == 'DELETE':
            body = ''
            return (httplib.NO_CONTENT, body, {}, httplib.responses[httplib.NO_CONTENT])

    def _api_2_0_balance(self, method, url, body, headers):
        body = self.fixtures.load('balance.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_2_0_pricing(self, method, url, body, headers):
        body = self.fixtures.load('pricing.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_2_0_currentusage(self, method, url, body, headers):
        body = self.fixtures.load('currentusage.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_2_0_subscriptions(self, method, url, body, headers):
        body = self.fixtures.load('subscriptions.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_2_0_subscriptions_STATUS_FILTER(self, method, url, body, headers):
        self.assertUrlContainsQueryParams(url, {'status': 'active'})
        body = self.fixtures.load('subscriptions.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_2_0_subscriptions_RESOURCE_FILTER(self, method, url, body, headers):
        expected_params = {'resource': 'cpu,mem', 'status': 'all'}
        self.assertUrlContainsQueryParams(url, expected_params)
        body = self.fixtures.load('subscriptions.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_2_0_subscriptions_7272_action_auto_renew(self, method, url, body, headers):
        body = ''
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_2_0_subscriptions_CREATE_SUBSCRIPTION(self, method, url, body, headers):
        body = self.fixtures.load('create_subscription.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_2_0_capabilities(self, method, url, body, headers):
        body = self.fixtures.load('capabilities.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_2_0_servers_availability_groups(self, method, url, body, headers):
        body = self.fixtures.load('servers_avail_groups.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_2_0_drives_availability_groups(self, method, url, body, headers):
        body = self.fixtures.load('drives_avail_groups.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_2_0_keypairs(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('keypairs_list.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        elif method == 'POST':
            body = self.fixtures.load('keypairs_import.json')
            return (httplib.CREATED, body, {}, httplib.responses[httplib.CREATED])

    def _api_2_0_keypairs_186106ac_afb5_40e5_a0de_6f0feba5a3d5(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('keypairs_get.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        if method == 'DELETE':
            body = ''
            return (httplib.NO_CONTENT, body, {}, httplib.responses[httplib.NO_CONTENT])