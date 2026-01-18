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
class OpenStack_1_1_MockHttp(MockHttp, unittest.TestCase):
    fixtures = ComputeFileFixtures('openstack_v1.1')
    auth_fixtures = OpenStackFixtures()
    json_content_headers = {'content-type': 'application/json; charset=UTF-8'}

    def _v2_0_tokens(self, method, url, body, headers):
        body = self.auth_fixtures.load('_v2_0__auth.json')
        return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v1_0(self, method, url, body, headers):
        headers = {'x-auth-token': 'FE011C19-CF86-4F87-BE5D-9229145D7A06', 'x-server-management-url': 'https://api.example.com/v1.1/slug'}
        return (httplib.NO_CONTENT, '', headers, httplib.responses[httplib.NO_CONTENT])

    def _v1_1_slug_servers_detail(self, method, url, body, headers):
        body = self.fixtures.load('_servers_detail.json')
        return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v1_1_slug_servers_detail_ERROR_STATE_NO_IMAGE_ID(self, method, url, body, headers):
        body = self.fixtures.load('_servers_detail_ERROR_STATE.json')
        return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v2_1337_servers_detail_UNAUTHORIZED(self, method, url, body, headers):
        return (httplib.UNAUTHORIZED, '', {}, httplib.responses[httplib.UNAUTHORIZED])

    def _v2_1337_servers_does_not_exist(self, *args, **kwargs):
        return (httplib.NOT_FOUND, None, {}, httplib.responses[httplib.NOT_FOUND])

    def _v1_1_slug_flavors_detail(self, method, url, body, headers):
        body = self.fixtures.load('_flavors_detail.json')
        return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v1_1_slug_images_detail(self, method, url, body, headers):
        body = self.fixtures.load('_images_detail.json')
        return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v1_1_slug_servers(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('_servers_create.json')
        elif method == 'GET':
            body = self.fixtures.load('_servers.json')
        else:
            raise NotImplementedError()
        return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v1_1_slug_servers_26f7fbee_8ce1_4c28_887a_bfe8e4bb10fe(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_servers_26f7fbee_8ce1_4c28_887a_bfe8e4bb10fe.json')
        else:
            raise NotImplementedError()
        return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v1_1_slug_servers_12065_action(self, method, url, body, headers):
        if method != 'POST':
            self.fail('HTTP method other than POST to action URL')
        return (httplib.ACCEPTED, '', {}, httplib.responses[httplib.ACCEPTED])

    def _v1_1_slug_servers_12064_action(self, method, url, body, headers):
        if method != 'POST':
            self.fail('HTTP method other than POST to action URL')
        if 'createImage' in json.loads(body):
            return (httplib.ACCEPTED, '', {'location': 'http://127.0.0.1/v1.1/68/images/4949f9ee-2421-4c81-8b49-13119446008b'}, httplib.responses[httplib.ACCEPTED])
        elif 'rescue' in json.loads(body):
            return (httplib.OK, '{"adminPass": "foo"}', {}, httplib.responses[httplib.OK])
        return (httplib.ACCEPTED, '', {}, httplib.responses[httplib.ACCEPTED])

    def _v1_1_slug_servers_12066_action(self, method, url, body, headers):
        if method != 'POST':
            self.fail('HTTP method other than POST to action URL')
        if 'rebuild' not in json.loads(body):
            self.fail('Did not get expected action (rebuild) in action URL')
        self.assertTrue('"OS-DCF:diskConfig": "MANUAL"' in body, msg='Manual disk configuration option was not specified in rebuild body: ' + body)
        return (httplib.ACCEPTED, '', {}, httplib.responses[httplib.ACCEPTED])

    def _v1_1_slug_servers_12065(self, method, url, body, headers):
        if method == 'DELETE':
            return (httplib.ACCEPTED, '', {}, httplib.responses[httplib.ACCEPTED])
        else:
            raise NotImplementedError()

    def _v1_1_slug_servers_12064(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_servers_12064.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        elif method == 'PUT':
            body = self.fixtures.load('_servers_12064_updated_name_bob.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        elif method == 'DELETE':
            return (httplib.ACCEPTED, '', {}, httplib.responses[httplib.ACCEPTED])
        else:
            raise NotImplementedError()

    def _v1_1_slug_servers_12062(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_servers_12064.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v1_1_slug_servers_12064247(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_servers_12064247.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v1_1_slug_servers_12063_metadata(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_servers_12063_metadata_two_keys.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        elif method == 'PUT':
            body = self.fixtures.load('_servers_12063_metadata_two_keys.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v1_1_slug_servers_EX_DISK_CONFIG(self, method, url, body, headers):
        if method == 'POST':
            body = u(body)
            self.assertTrue(body.find('"OS-DCF:diskConfig": "AUTO"'))
            body = self.fixtures.load('_servers_create_disk_config.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v1_1_slug_servers_EX_FILES(self, method, url, body, headers):
        if method == 'POST':
            body = u(body)
            personality = [{'path': '/file1', 'contents': 'Y29udGVudDE='}, {'path': '/file2', 'contents': 'Y29udGVudDI='}]
            self.assertEqual(json.loads(body)['server']['personality'], personality)
            body = self.fixtures.load('_servers_create.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v1_1_slug_servers_EX_FILES_NONE(self, method, url, body, headers):
        if method == 'POST':
            body = u(body)
            self.assertNotIn('"personality"', body)
            body = self.fixtures.load('_servers_create.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v1_1_slug_flavors_7(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_flavors_7.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        else:
            raise NotImplementedError()

    def _v1_1_slug_images_13(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_images_13.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        else:
            raise NotImplementedError()

    def _v2_1337_v2_images_f24a3c1b_d52a_4116_91da_25b3eee8f55e(self, method, url, body, headers):
        if method == 'GET' or method == 'PATCH':
            body = self.fixtures.load('_images_f24a3c1b-d52a-4116-91da-25b3eee8f55e.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        else:
            raise NotImplementedError()

    def _v2_1337_v2_images_d9a9cd9a_278a_444c_90a6_d24b8c688a63_members(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_images_d9a9cd9a_278a_444c_90a6_d24b8c688a63_members.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        else:
            raise NotImplementedError()

    def _v2_1337_v2_images_9af1a54e_a1b2_4df8_b747_4bec97abc799_members(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('_images_9af1a54e_a1b2_4df8_b747_4bec97abc799_members.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        else:
            raise NotImplementedError()

    def _v2_1337_v2_images_8af1a54e_a1b2_4df8_b747_4bec97abc799_members_e2151b1fe02d4a8a2d1f5fc331522c0a(self, method, url, body, headers):
        if method == 'PUT':
            body = self.fixtures.load('_images_8af1a54e_a1b2_4df8_b747_4bec97abc799_members.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        else:
            raise NotImplementedError()

    def _v2_1337_v2_images_d9a9cd9a_278a_444c_90a6_d24b8c688a63_members_016926dff12345e8b10329f24c99745b(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_images_d9a9cd9a_278a_444c_90a6_d24b8c688a63_members_016926dff12345e8b10329f24c99745b.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        else:
            raise NotImplementedError()

    def _v2_1337_v2_images(self, method, url, body, headers):
        if method == 'GET':
            if 'marker=e7a40226-3523-4f0f-87d8-d8dc91bbf4a3' in url:
                body = self.fixtures.load('_images_v2_page2.json')
            else:
                body = self.fixtures.load('_images_v2.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        else:
            raise NotImplementedError()

    def _v2_1337_v2_images_invalid_next(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_images_v2_invalid_next.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        else:
            raise NotImplementedError()

    def _v1_1_slug_images_26365521_8c62_11f9_2c33_283d153ecc3a(self, method, url, body, headers):
        if method == 'DELETE':
            return (httplib.NO_CONTENT, '', {}, httplib.responses[httplib.NO_CONTENT])
        else:
            raise NotImplementedError()

    def _v1_1_slug_images_4949f9ee_2421_4c81_8b49_13119446008b(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_images_4949f9ee_2421_4c81_8b49_13119446008b.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        else:
            raise NotImplementedError()

    def _v2_1337_v2_images_4949f9ee_2421_4c81_8b49_13119446008b(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_images_f24a3c1b-d52a-4116-91da-25b3eee8f55d.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        else:
            raise NotImplementedError()

    def _v2_1337_v2_0_ports(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_ports_v2.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        elif method == 'POST':
            body = self.fixtures.load('_port_v2.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        else:
            raise NotImplementedError()

    def _v2_1337_v2_0_ports_126da55e_cfcb_41c8_ae39_a26cb8a7e723(self, method, url, body, headers):
        if method == 'DELETE':
            return (httplib.NO_CONTENT, '', {}, httplib.responses[httplib.NO_CONTENT])
        elif method == 'GET':
            body = self.fixtures.load('_port_v2.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        elif method == 'PUT':
            if body:
                body = self.fixtures.load('_port_v2.json')
                return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
            else:
                return (httplib.INTERNAL_SERVER_ERROR, '', {}, httplib.responses[httplib.INTERNAL_SERVER_ERROR])
        else:
            raise NotImplementedError()

    def _v2_1337_servers_12065_os_volume_attachments_DEVICE_AUTO(self, method, url, body, headers):
        if method == 'POST':
            if 'rackspace' not in self.__class__.__name__.lower():
                body = json.loads(body)
                self.assertEqual(body['volumeAttachment']['device'], None)
            return (httplib.NO_CONTENT, '', {}, httplib.responses[httplib.NO_CONTENT])
        else:
            raise NotImplementedError()

    def _v2_1337_servers_1c01300f_ef97_4937_8f03_ac676d6234be_os_interface_126da55e_cfcb_41c8_ae39_a26cb8a7e723(self, method, url, body, headers):
        if method == 'DELETE':
            return (httplib.NO_CONTENT, '', {}, httplib.responses[httplib.NO_CONTENT])
        else:
            raise NotImplementedError()

    def _v2_1337_servers_1c01300f_ef97_4937_8f03_ac676d6234be_os_interface(self, method, url, body, headers):
        if method == 'POST':
            return (httplib.NO_CONTENT, '', {}, httplib.responses[httplib.NO_CONTENT])
        else:
            raise NotImplementedError()

    def _v2_1337_servers_26f7fbee_8ce1_4c28_887a_bfe8e4bb10fe_EX_FILES(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_servers_26f7fbee_8ce1_4c28_887a_bfe8e4bb10fe.json')
        else:
            raise NotImplementedError()
        return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v2_1337_servers_26f7fbee_8ce1_4c28_887a_bfe8e4bb10fe_EX_FILES_NONE(self, method, url, body, headers):
        return self._v2_1337_servers_26f7fbee_8ce1_4c28_887a_bfe8e4bb10fe_EX_FILES(method, url, body, headers)

    def _v1_1_slug_servers_1c01300f_ef97_4937_8f03_ac676d6234be_os_security_groups(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_servers_1c01300f-ef97-4937-8f03-ac676d6234be_os-security-groups.json')
        else:
            raise NotImplementedError()
        return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v1_1_slug_os_security_groups(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_os_security_groups.json')
        elif method == 'POST':
            body = self.fixtures.load('_os_security_groups_create.json')
        else:
            raise NotImplementedError()
        return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v1_1_slug_os_security_groups_6(self, method, url, body, headers):
        if method == 'DELETE':
            return (httplib.NO_CONTENT, '', {}, httplib.responses[httplib.NO_CONTENT])
        else:
            raise NotImplementedError()

    def _v1_1_slug_os_security_group_rules(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('_os_security_group_rules_create.json')
        else:
            raise NotImplementedError()
        return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v1_1_slug_os_security_group_rules_2(self, method, url, body, headers):
        if method == 'DELETE':
            return (httplib.NO_CONTENT, '', {}, httplib.responses[httplib.NO_CONTENT])
        else:
            raise NotImplementedError()

    def _v1_1_slug_os_keypairs(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_os_keypairs.json')
        elif method == 'POST':
            if 'public_key' in body:
                body = self.fixtures.load('_os_keypairs_create_import.json')
            else:
                body = self.fixtures.load('_os_keypairs_create.json')
        else:
            raise NotImplementedError()
        return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v1_1_slug_os_keypairs_test_key_pair(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_os_keypairs_get_one.json')
        else:
            raise NotImplementedError()
        return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v1_1_slug_os_keypairs_doesnt_exist(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_os_keypairs_not_found.json')
        else:
            raise NotImplementedError()
        return (httplib.NOT_FOUND, body, self.json_content_headers, httplib.responses[httplib.NOT_FOUND])

    def _v1_1_slug_os_keypairs_key1(self, method, url, body, headers):
        if method == 'DELETE':
            return (httplib.ACCEPTED, '', {}, httplib.responses[httplib.ACCEPTED])
        else:
            raise NotImplementedError()

    def _v1_1_slug_os_volumes(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_os_volumes.json')
        elif method == 'POST':
            body = self.fixtures.load('_os_volumes_create.json')
        else:
            raise NotImplementedError()
        return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v1_1_slug_os_volumes_cd76a3a1_c4ce_40f6_9b9f_07a61508938d(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_os_volumes_cd76a3a1_c4ce_40f6_9b9f_07a61508938d.json')
        elif method == 'DELETE':
            body = ''
        else:
            raise NotImplementedError()
        return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v1_1_slug_servers_12065_os_volume_attachments(self, method, url, body, headers):
        if method == 'POST':
            if 'rackspace' not in self.__class__.__name__.lower():
                body = json.loads(body)
                self.assertEqual(body['volumeAttachment']['device'], '/dev/sdb')
            body = self.fixtures.load('_servers_12065_os_volume_attachments.json')
        else:
            raise NotImplementedError()
        return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v1_1_slug_servers_12065_os_volume_attachments_cd76a3a1_c4ce_40f6_9b9f_07a61508938d(self, method, url, body, headers):
        if method == 'DELETE':
            body = ''
        else:
            raise NotImplementedError()
        return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v1_1_slug_os_floating_ip_pools(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_floating_ip_pools.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        else:
            raise NotImplementedError()

    def _v1_1_slug_os_floating_ips_foo_bar_id(self, method, url, body, headers):
        if method == 'DELETE':
            body = ''
            return (httplib.ACCEPTED, body, self.json_content_headers, httplib.responses[httplib.OK])
        else:
            raise NotImplementedError()

    def _v1_1_slug_os_floating_ips(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_floating_ips.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        elif method == 'POST':
            body = self.fixtures.load('_floating_ip.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        else:
            raise NotImplementedError()

    def _v1_1_slug_servers_4242_action(self, method, url, body, headers):
        if method == 'POST':
            body = ''
            return (httplib.ACCEPTED, body, self.json_content_headers, httplib.responses[httplib.OK])
        else:
            raise NotImplementedError()

    def _v1_1_slug_os_networks(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_os_networks.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        elif method == 'POST':
            body = self.fixtures.load('_os_networks_POST.json')
            return (httplib.ACCEPTED, body, self.json_content_headers, httplib.responses[httplib.OK])
        raise NotImplementedError()

    def _v1_1_slug_os_networks_f13e5051_feea_416b_827a_1a0acc2dad14(self, method, url, body, headers):
        if method == 'DELETE':
            body = ''
            return (httplib.ACCEPTED, body, self.json_content_headers, httplib.responses[httplib.OK])
        raise NotImplementedError()

    def _v1_1_slug_servers_72258_action(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('_servers_suspend.json')
            return (httplib.ACCEPTED, body, self.json_content_headers, httplib.responses[httplib.OK])
        else:
            raise NotImplementedError()

    def _v1_1_slug_servers_12063_action(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('_servers_unpause.json')
            return (httplib.ACCEPTED, body, self.json_content_headers, httplib.responses[httplib.OK])
        else:
            raise NotImplementedError()

    def _v1_1_slug_servers_12086_action(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('_servers_12086_console_output.json')
            return (httplib.ACCEPTED, body, self.json_content_headers, httplib.responses[httplib.OK])
        else:
            raise NotImplementedError()

    def _v1_1_slug_os_snapshots(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_os_snapshots.json')
        elif method == 'POST':
            body = self.fixtures.load('_os_snapshots_create.json')
        else:
            raise NotImplementedError()
        return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v1_1_slug_os_snapshots_3fbbcccf_d058_4502_8844_6feeffdf4cb5(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_os_snapshot.json')
            status_code = httplib.OK
        elif method == 'DELETE':
            body = ''
            status_code = httplib.NO_CONTENT
        else:
            raise NotImplementedError()
        return (status_code, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v1_1_slug_os_snapshots_3fbbcccf_d058_4502_8844_6feeffdf4cb5_RACKSPACE(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_os_snapshot_rackspace.json')
            status_code = httplib.OK
        elif method == 'DELETE':
            body = ''
            status_code = httplib.NO_CONTENT
        else:
            raise NotImplementedError()
        return (status_code, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v1_1_slug_os_snapshots_RACKSPACE(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_os_snapshots_rackspace.json')
        elif method == 'POST':
            body = self.fixtures.load('_os_snapshots_create_rackspace.json')
        else:
            raise NotImplementedError()
        return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v2_1337_v2_0_networks(self, method, url, body, headers):
        if method == 'GET':
            if 'router:external=True' in url:
                body = self.fixtures.load('_v2_0__networks_public.json')
                return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
            else:
                body = self.fixtures.load('_v2_0__networks.json')
                return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        elif method == 'POST':
            body = self.fixtures.load('_v2_0__networks_POST.json')
            return (httplib.ACCEPTED, body, self.json_content_headers, httplib.responses[httplib.OK])
        raise NotImplementedError()

    def _v2_1337_v2_0_networks_cc2dad14_827a_feea_416b_f13e50511a0a(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_v2_0__network.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        raise NotImplementedError()

    def _v2_1337_v2_0_networks_d32019d3_bc6e_4319_9c1d_6722fc136a22(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_v2_0__networks_POST.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        elif method == 'DELETE':
            body = ''
            return (httplib.NO_CONTENT, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v2_1337_v2_0_subnets_08eae331_0402_425a_923c_34f7cfe39c1b(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_v2_0__subnet.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        if method == 'DELETE':
            body = ''
            return (httplib.NO_CONTENT, body, self.json_content_headers, httplib.responses[httplib.OK])
        elif method == 'PUT':
            body = self.fixtures.load('_v2_0__subnet.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v2_1337_v2_0_subnets(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('_v2_0__subnet.json')
            return (httplib.CREATED, body, self.json_content_headers, httplib.responses[httplib.OK])
        else:
            body = self.fixtures.load('_v2_0__subnets.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v3_1337_volumes_detail(self, method, url, body, headers):
        body = self.fixtures.load('_v2_0__volumes.json')
        return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v3_1337_volumes(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('_v2_0__volume.json')
            return (httplib.CREATED, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v3_1337_volumes_cd76a3a1_c4ce_40f6_9b9f_07a61508938d(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_v2_0__volume.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        if method == 'DELETE':
            body = ''
            return (httplib.NO_CONTENT, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v3_1337_volumes_abc6a3a1_c4ce_40f6_9b9f_07a61508938d(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_v2_0__volume_abc6a3a1_c4ce_40f6_9b9f_07a61508938d.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        if method == 'DELETE':
            body = ''
            return (httplib.NO_CONTENT, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v3_1337_snapshots_detail(self, method, url, body, headers):
        if 'unit_test=paginate' in url and 'marker' not in url or 'unit_test=pagination_loop' in url:
            body = self.fixtures.load('_v2_0__snapshots_paginate_start.json')
        else:
            body = self.fixtures.load('_v2_0__snapshots.json')
        return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v3_1337_snapshots(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('_v2_0__snapshot.json')
            return (httplib.CREATED, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v3_1337_snapshots_3fbbcccf_d058_4502_8844_6feeffdf4cb5(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_v2_0__snapshot.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        if method == 'DELETE':
            body = ''
            return (httplib.NO_CONTENT, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v2_1337_v2_0_security_groups(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('_v2_0__security_group.json')
            return (httplib.CREATED, body, self.json_content_headers, httplib.responses[httplib.OK])
        if method == 'GET':
            body = self.fixtures.load('_v2_0__security_groups.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v2_1337_v2_0_security_groups_6(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_v2_0__security_group.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        if method == 'DELETE':
            body = ''
            return (httplib.NO_CONTENT, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v2_1337_v2_0_security_group_rules(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('_v2_0__security_group_rule.json')
            return (httplib.CREATED, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v2_1337_v2_0_security_group_rules_2(self, method, url, body, headers):
        if method == 'DELETE':
            body = ''
            return (httplib.NO_CONTENT, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v2_1337_v2_0_floatingips(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('_v2_0__floatingip.json')
            return (httplib.CREATED, body, self.json_content_headers, httplib.responses[httplib.OK])
        if method == 'GET':
            if 'floating_network_id=' in url:
                body = self.fixtures.load('_v2_0__floatingips_net_id.json')
            elif 'floating_ip_address' in url:
                body = self.fixtures.load('_v2_0__floatingips_ip_id.json')
            else:
                body = self.fixtures.load('_v2_0__floatingips.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v2_1337_v2_0_floatingips_foo_bar_id(self, method, url, body, headers):
        if method == 'DELETE':
            body = ''
            return (httplib.NO_CONTENT, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v2_1337_v2_0_floatingips_09ea1784_2f81_46dc_8c91_244b4df75bde(self, method, url, body, headers):
        if method == 'PUT':
            self.assertIn(body, ['{"floatingip": {"port_id": "ce531f90-199f-48c0-816c-13e38010b442"}}', '{"floatingip": {"port_id": null}}'])
            body = ''
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v2_1337_v2_0_routers_f8a44de0_fc8e_45df_93c7_f79bf3b01c95(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_v2_0__router.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        if method == 'DELETE':
            body = ''
            return (httplib.NO_CONTENT, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v2_1337_v2_0_routers(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('_v2_0__router.json')
            return (httplib.CREATED, body, self.json_content_headers, httplib.responses[httplib.OK])
        else:
            body = self.fixtures.load('_v2_0__routers.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v2_1337_v2_0_routers_f8a44de0_fc8e_45df_93c7_f79bf3b01c95_add_router_interface(self, method, url, body, headers):
        if method == 'PUT':
            body = self.fixtures.load('_v2_0__router_interface.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v2_1337_v2_0_routers_f8a44de0_fc8e_45df_93c7_f79bf3b01c95_remove_router_interface(self, method, url, body, headers):
        if method == 'PUT':
            body = self.fixtures.load('_v2_0__router_interface.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v2_1337_os_quota_sets_tenant_id_detail(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_v2_0__quota_set.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v2_1337_flavors_7_os_extra_specs(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_flavor_extra_specs.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        else:
            raise NotImplementedError()

    def _v2_1337_servers_1000_action(self, method, url, body, headers):
        if method != 'POST' or body != '{"removeSecurityGroup": {"name": "sgname"}}':
            raise NotImplementedError(body)
        return (httplib.ACCEPTED, None, {}, httplib.responses[httplib.ACCEPTED])

    def _v2_1337_v2_0_quotas_tenant_id_details_json(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_v2_0__network_quota.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v3_1337_os_quota_sets_tenant_id(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_v3_0__volume_quota.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v2_1337_os_server_groups(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_v2_0__os_server_groups.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        elif method == 'POST':
            body = self.fixtures.load('_v2_0__os_server_group.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v2_1337_os_server_groups_616fb98f_46ca_475e_917e_2563e5a8cd19(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_v2_0__os_server_group.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        elif method == 'DELETE':
            body = ''
            return (httplib.NO_CONTENT, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v2_1337_servers_4242_os_interface(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_servers_os_intefaces.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])