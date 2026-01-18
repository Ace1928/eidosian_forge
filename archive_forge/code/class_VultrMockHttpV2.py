import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.common.vultr import VultrException
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vultr import VultrNodeDriver, VultrNodeDriverV2
class VultrMockHttpV2(MockHttp):
    fixtures = ComputeFileFixtures('vultr_v2')

    def _v2_regions(self, method, url, body, headers):
        body = self.fixtures.load('list_locations.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v2_os(self, method, url, body, headers):
        body = self.fixtures.load('list_images.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v2_plans(self, method, url, body, headers):
        body = self.fixtures.load('list_sizes.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v2_instances(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('list_nodes.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        elif method == 'POST':
            body = self.fixtures.load('create_node.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v2_bare_metals(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('ex_list_bare_metal_nodes.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        elif method == 'POST':
            body = self.fixtures.load('create_node_bare_metal.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v2_instances_123(self, method, url, body, headers):
        if method == 'DELETE':
            return (httplib.NO_CONTENT, body, {}, httplib.responses[httplib.NO_CONTENT])
        elif method == 'GET':
            body = self.fixtures.load('ex_get_node.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        elif method == 'PATCH':
            body = self.fixtures.load('ex_resize_node.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v2_instances_123_start(self, method, url, body, headers):
        return (httplib.NO_CONTENT, body, {}, httplib.responses[httplib.NO_CONTENT])

    def _v2_instances_123_reboot(self, method, url, body, headers):
        return (httplib.NO_CONTENT, body, {}, httplib.responses[httplib.NO_CONTENT])

    def _v2_instances_halt(self, method, url, body, headers):
        return (httplib.NO_CONTENT, body, {}, httplib.responses[httplib.NO_CONTENT])

    def _v2_ssh_keys(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('list_key_pairs.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        elif method == 'POST':
            body = self.fixtures.load('import_key_pair_from_string.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v2_ssh_keys_UNAUTHORIZED(self, method, url, body, headers):
        body = '{"error":"Invalid API token.","status":401}'
        return (httplib.UNAUTHORIZED, body, {}, httplib.responses[httplib.UNAUTHORIZED])

    def _v2_ssh_keys_123(self, method, url, body, headers):
        if method == 'DELETE':
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        elif method == 'GET':
            body = self.fixtures.load('get_key_pair.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v2_blocks(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('list_volumes.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        elif method == 'POST':
            body = self.fixtures.load('create_volume.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v2_blocks_123(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('ex_get_volume.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        elif method == 'DELETE':
            return (httplib.NO_CONTENT, body, {}, httplib.responses[httplib.NO_CONTENT])
        elif method == 'PATCH':
            return (httplib.NO_CONTENT, body, {}, httplib.responses[httplib.NO_CONTENT])

    def _v2_blocks_123_attach(self, method, url, body, headers):
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v2_blocks_1234_attach_WRONG_LOCATION(self, method, url, body, headers):
        body = '{"error": "unable to attach: Block storage volume must be in the same region as the server it is attached to.", "status": 400}'
        return (httplib.BAD_REQUEST, body, {}, httplib.responses[httplib.BAD_REQUEST])

    def _v2_blocks_123_detach(self, method, url, body, headers):
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v2_regions_ams_availability(self, method, url, body, headers):
        body = self.fixtures.load('ex_list_available_sizes_for_location.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v2_private_networks(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('ex_list_networks.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        elif method == 'POST':
            body = self.fixtures.load('ex_create_network.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v2_private_networks_123(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('ex_get_network.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        elif method == 'DELETE':
            return (httplib.NO_CONTENT, body, {}, httplib.responses[httplib.NO_CONTENT])

    def _v2_os_PAGINATED(self, method, url, body, headers):
        if 'cursor' not in url:
            body = self.fixtures.load('list_images_paginated_1.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        elif 'cursor=bmV4dF9fMjMw' in url:
            body = self.fixtures.load('list_images_paginated_2.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        else:
            body = self.fixtures.load('list_images_paginated_3.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v2_snapshots(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('ex_list_snapshots.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        elif method == 'POST':
            body = self.fixtures.load('ex_create_snapshot.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v2_snapshots_123(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('ex_get_snapshot.json')
            return (httplib.OK, body, {}, httplib.responses[httplib.OK])
        elif method == 'DELETE':
            return (httplib.NO_CONTENT, body, {}, httplib.responses[httplib.NO_CONTENT])

    def _v2_bare_metals_234_reboot(self, method, url, body, headers):
        return (httplib.NO_CONTENT, body, {}, httplib.responses[httplib.NO_CONTENT])

    def _v2_bare_metals_234_start(self, method, url, body, headers):
        return (httplib.NO_CONTENT, body, {}, httplib.responses[httplib.NO_CONTENT])

    def _v2_bare_metals_234_halt(self, method, url, body, headers):
        return (httplib.NO_CONTENT, body, {}, httplib.responses[httplib.NO_CONTENT])

    def _v2_bare_metals_234(self, method, url, body, headers):
        return (httplib.NO_CONTENT, body, {}, httplib.responses[httplib.NO_CONTENT])

    def _v2_plans_metal(self, method, url, body, headers):
        body = self.fixtures.load('ex_list_bare_metal_sizes.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])