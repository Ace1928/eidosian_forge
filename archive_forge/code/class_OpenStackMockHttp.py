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
class OpenStackMockHttp(MockHttp, unittest.TestCase):
    fixtures = ComputeFileFixtures('openstack')
    auth_fixtures = OpenStackFixtures()
    json_content_headers = {'content-type': 'application/json; charset=UTF-8'}

    def _v1_0(self, method, url, body, headers):
        headers = {'x-server-management-url': 'https://servers.api.rackspacecloud.com/v1.0/slug', 'x-auth-token': 'FE011C19-CF86-4F87-BE5D-9229145D7A06', 'x-cdn-management-url': 'https://cdn.clouddrive.com/v1/MossoCloudFS_FE011C19-CF86-4F87-BE5D-9229145D7A06', 'x-storage-token': 'FE011C19-CF86-4F87-BE5D-9229145D7A06', 'x-storage-url': 'https://storage4.clouddrive.com/v1/MossoCloudFS_FE011C19-CF86-4F87-BE5D-9229145D7A06'}
        return (httplib.NO_CONTENT, '', headers, httplib.responses[httplib.NO_CONTENT])

    def _v1_0_UNAUTHORIZED(self, method, url, body, headers):
        return (httplib.UNAUTHORIZED, '', {}, httplib.responses[httplib.UNAUTHORIZED])

    def _v1_0_INTERNAL_SERVER_ERROR(self, method, url, body, headers):
        return (httplib.INTERNAL_SERVER_ERROR, '<h1>500: Internal Server Error</h1>', {}, httplib.responses[httplib.INTERNAL_SERVER_ERROR])

    def _v1_0_slug_images_detail_NO_MESSAGE_IN_ERROR_BODY(self, method, url, body, headers):
        body = self.fixtures.load('300_multiple_choices.json')
        return (httplib.MULTIPLE_CHOICES, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v1_0_UNAUTHORIZED_MISSING_KEY(self, method, url, body, headers):
        headers = {'x-server-management-url': 'https://servers.api.rackspacecloud.com/v1.0/slug', 'x-auth-tokenx': 'FE011C19-CF86-4F87-BE5D-9229145D7A06', 'x-cdn-management-url': 'https://cdn.clouddrive.com/v1/MossoCloudFS_FE011C19-CF86-4F87-BE5D-9229145D7A06'}
        return (httplib.NO_CONTENT, '', headers, httplib.responses[httplib.NO_CONTENT])

    def _v2_0_tokens(self, method, url, body, headers):
        body = self.auth_fixtures.load('_v2_0__auth.json')
        return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v1_0_slug_servers_detail_EMPTY(self, method, url, body, headers):
        body = self.fixtures.load('v1_slug_servers_detail_empty.xml')
        return (httplib.OK, body, XML_HEADERS, httplib.responses[httplib.OK])

    def _v1_0_slug_servers_detail(self, method, url, body, headers):
        body = self.fixtures.load('v1_slug_servers_detail.xml')
        return (httplib.OK, body, XML_HEADERS, httplib.responses[httplib.OK])

    def _v1_0_slug_servers_detail_METADATA(self, method, url, body, headers):
        body = self.fixtures.load('v1_slug_servers_detail_metadata.xml')
        return (httplib.OK, body, XML_HEADERS, httplib.responses[httplib.OK])

    def _v1_0_slug_servers_detail_UNAUTHORIZED(self, method, url, body, headers):
        return (httplib.UNAUTHORIZED, '', {}, httplib.responses[httplib.UNAUTHORIZED])

    def _v1_0_slug_images_333111(self, method, url, body, headers):
        if method != 'DELETE':
            raise NotImplementedError()
        return (httplib.NO_CONTENT, '', {}, httplib.responses[httplib.NO_CONTENT])

    def _v1_0_slug_images(self, method, url, body, headers):
        if method != 'POST':
            raise NotImplementedError()
        body = self.fixtures.load('v1_slug_images_post.xml')
        return (httplib.ACCEPTED, body, XML_HEADERS, httplib.responses[httplib.ACCEPTED])

    def _v1_0_slug_images_detail(self, method, url, body, headers):
        if method != 'GET':
            raise ValueError('Invalid method: %s' % method)
        body = self.fixtures.load('v1_slug_images_detail.xml')
        return (httplib.OK, body, XML_HEADERS, httplib.responses[httplib.OK])

    def _v1_0_slug_images_detail_invalid_next(self, method, url, body, headers):
        if method != 'GET':
            raise ValueError('Invalid method: %s' % method)
        body = self.fixtures.load('v1_slug_images_detail.xml')
        return (httplib.OK, body, XML_HEADERS, httplib.responses[httplib.OK])

    def _v1_0_slug_servers(self, method, url, body, headers):
        body = self.fixtures.load('v1_slug_servers.xml')
        return (httplib.ACCEPTED, body, XML_HEADERS, httplib.responses[httplib.ACCEPTED])

    def _v1_0_slug_servers_NO_ADMIN_PASS(self, method, url, body, headers):
        body = self.fixtures.load('v1_slug_servers_no_admin_pass.xml')
        return (httplib.ACCEPTED, body, XML_HEADERS, httplib.responses[httplib.ACCEPTED])

    def _v1_0_slug_servers_EX_SHARED_IP_GROUP(self, method, url, body, headers):
        body = u(body)
        self.assertTrue(body.find('sharedIpGroupId="12345"') != -1)
        body = self.fixtures.load('v1_slug_servers.xml')
        return (httplib.ACCEPTED, body, XML_HEADERS, httplib.responses[httplib.ACCEPTED])

    def _v1_0_slug_servers_METADATA(self, method, url, body, headers):
        body = self.fixtures.load('v1_slug_servers_metadata.xml')
        return (httplib.ACCEPTED, body, XML_HEADERS, httplib.responses[httplib.ACCEPTED])

    def _v1_0_slug_servers_72258_action(self, method, url, body, headers):
        if method != 'POST' or body[:8] != '<reboot ':
            raise NotImplementedError()
        return (httplib.ACCEPTED, '', {}, httplib.responses[httplib.ACCEPTED])

    def _v1_0_slug_limits(self, method, url, body, headers):
        body = self.fixtures.load('v1_slug_limits.xml')
        return (httplib.ACCEPTED, body, XML_HEADERS, httplib.responses[httplib.ACCEPTED])

    def _v1_0_slug_servers_72258(self, method, url, body, headers):
        if method != 'DELETE':
            raise NotImplementedError()
        return (httplib.ACCEPTED, '', {}, httplib.responses[httplib.ACCEPTED])

    def _v1_0_slug_servers_72258_ips(self, method, url, body, headers):
        body = self.fixtures.load('v1_slug_servers_ips.xml')
        return (httplib.OK, body, XML_HEADERS, httplib.responses[httplib.OK])

    def _v1_0_slug_shared_ip_groups_5467(self, method, url, body, headers):
        if method != 'DELETE':
            raise NotImplementedError()
        return (httplib.NO_CONTENT, '', {}, httplib.responses[httplib.NO_CONTENT])

    def _v1_0_slug_shared_ip_groups(self, method, url, body, headers):
        fixture = 'v1_slug_shared_ip_group.xml' if method == 'POST' else 'v1_slug_shared_ip_groups.xml'
        body = self.fixtures.load(fixture)
        return (httplib.OK, body, XML_HEADERS, httplib.responses[httplib.OK])

    def _v1_0_slug_shared_ip_groups_detail(self, method, url, body, headers):
        body = self.fixtures.load('v1_slug_shared_ip_groups_detail.xml')
        return (httplib.OK, body, XML_HEADERS, httplib.responses[httplib.OK])

    def _v1_0_slug_servers_3445_ips_public_67_23_21_133(self, method, url, body, headers):
        return (httplib.ACCEPTED, '', {}, httplib.responses[httplib.ACCEPTED])

    def _v1_0_slug_servers_444222_action(self, method, url, body, headers):
        body = u(body)
        if body.find('resize') != -1:
            if body.find('personality') != -1:
                return httplib.BAD_REQUEST
            else:
                return (httplib.ACCEPTED, '', headers, httplib.responses[httplib.NO_CONTENT])
        elif body.find('confirmResize') != -1:
            return (httplib.NO_CONTENT, '', headers, httplib.responses[httplib.NO_CONTENT])
        elif body.find('revertResize') != -1:
            return (httplib.NO_CONTENT, '', headers, httplib.responses[httplib.NO_CONTENT])

    def _v1_0_slug_flavors_detail(self, method, url, body, headers):
        body = self.fixtures.load('v1_slug_flavors_detail.xml')
        headers = {'date': 'Tue, 14 Jun 2011 09:43:55 GMT', 'content-length': '529'}
        headers.update(XML_HEADERS)
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _v1_1_auth(self, method, url, body, headers):
        body = self.auth_fixtures.load('_v1_1__auth.json')
        return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v1_1_auth_UNAUTHORIZED(self, method, url, body, headers):
        body = self.auth_fixtures.load('_v1_1__auth_unauthorized.json')
        return (httplib.UNAUTHORIZED, body, self.json_content_headers, httplib.responses[httplib.UNAUTHORIZED])

    def _v1_1_auth_UNAUTHORIZED_MISSING_KEY(self, method, url, body, headers):
        body = self.auth_fixtures.load('_v1_1__auth_mssing_token.json')
        return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])

    def _v1_1_auth_INTERNAL_SERVER_ERROR(self, method, url, body, headers):
        return (httplib.INTERNAL_SERVER_ERROR, '<h1>500: Internal Server Error</h1>', {'content-type': 'text/html'}, httplib.responses[httplib.INTERNAL_SERVER_ERROR])