import sys
import unittest
from datetime import datetime
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.compute.base import NodeImage
from libcloud.test.secrets import SCALEWAY_PARAMS
from libcloud.utils.iso8601 import UTC
from libcloud.common.exceptions import BaseHTTPError
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.scaleway import ScalewayNodeDriver
class ScalewayMockHttp(MockHttp):
    fixtures = ComputeFileFixtures('scaleway')

    def _products_servers(self, method, url, body, headers):
        body = self.fixtures.load('list_sizes.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _products_servers_availability(self, method, url, body, headers):
        body = self.fixtures.load('list_availability.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _servers_UNAUTHORIZED(self, method, url, body, headers):
        body = self.fixtures.load('error.json')
        return (httplib.UNAUTHORIZED, body, {}, httplib.responses[httplib.UNAUTHORIZED])

    def _images(self, method, url, body, headers):
        body = self.fixtures.load('list_images.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _images_POST(self, method, url, body, headers):
        body = self.fixtures.load('create_image.json')
        return (httplib.CREATED, body, {}, httplib.responses[httplib.CREATED])

    def _images_12345_DELETE(self, method, url, body, headers):
        return (httplib.NO_CONTENT, body, {}, httplib.responses[httplib.NO_CONTENT])

    def _images_12345(self, method, url, body, headers):
        body = self.fixtures.load('get_image.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _servers(self, method, url, body, headers):
        body = self.fixtures.load('list_nodes.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _servers_POST(self, method, url, body, headers):
        body = self.fixtures.load('create_node.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _servers_741db378_action_POST(self, method, url, body, headers):
        return (httplib.NO_CONTENT, body, {}, httplib.responses[httplib.NO_CONTENT])

    def _servers_INVALID_IMAGE(self, method, url, body, headers):
        body = self.fixtures.load('error_invalid_image.json')
        return (httplib.NOT_FOUND, body, {}, httplib.responses[httplib.NOT_FOUND])

    def _servers_741db378_action_REBOOT(self, method, url, body, headers):
        body = self.fixtures.load('reboot_node.json')
        return (httplib.CREATED, body, {}, httplib.responses[httplib.CREATED])

    def _servers_741db378_action_TERMINATE(self, method, url, body, headers):
        return (httplib.NO_CONTENT, body, {}, httplib.responses[httplib.NO_CONTENT])

    def _volumes(self, method, url, body, headers):
        body = self.fixtures.load('list_volumes.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _volumes_EMPTY(self, method, url, body, headers):
        body = self.fixtures.load('list_volumes_empty.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _snapshots(self, method, url, body, headers):
        body = self.fixtures.load('list_volume_snapshots.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _volumes_POST(self, method, url, body, headers):
        body = self.fixtures.load('create_volume.json')
        return (httplib.CREATED, body, {}, httplib.responses[httplib.CREATED])

    def _snapshots_POST(self, method, url, body, headers):
        body = self.fixtures.load('create_volume_snapshot.json')
        return (httplib.CREATED, body, {}, httplib.responses[httplib.CREATED])

    def _volumes_f929fe39_63f8_4be8_a80e_1e9c8ae22a76_DELETE(self, method, url, body, headers):
        return (httplib.NO_CONTENT, None, {}, httplib.responses[httplib.NO_CONTENT])

    def _snapshots_6f418e5f_b42d_4423_a0b5_349c74c454a4_DELETE(self, method, url, body, headers):
        return (httplib.NO_CONTENT, None, {}, httplib.responses[httplib.NO_CONTENT])

    def _tokens_token(self, method, url, body, headers):
        body = self.fixtures.load('token_info.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _users_5bea0358(self, method, url, body, headers):
        body = self.fixtures.load('user_info.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])