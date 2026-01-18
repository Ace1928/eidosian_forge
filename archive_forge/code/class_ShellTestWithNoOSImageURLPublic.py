import argparse
from collections import OrderedDict
import hashlib
import io
import logging
import os
import sys
import traceback
from unittest import mock
import uuid
import fixtures
from keystoneauth1 import exceptions as ks_exc
from keystoneauth1 import fixture as ks_fixture
from requests_mock.contrib import fixture as rm_fixture
from glanceclient.common import utils
from glanceclient import exc
from glanceclient import shell as openstack_shell
from glanceclient.tests.unit.v2.fixtures import image_show_fixture
from glanceclient.tests.unit.v2.fixtures import image_versions_fixture
from glanceclient.tests import utils as testutils
from glanceclient.v2 import schemas as schemas
import json
class ShellTestWithNoOSImageURLPublic(ShellTestWithKeystoneV3Auth):
    auth_env = FAKE_V4_ENV.copy()

    def setUp(self):
        super(ShellTestWithNoOSImageURLPublic, self).setUp()
        self.image_url = DEFAULT_IMAGE_URL
        self.requests.get(DEFAULT_IMAGE_URL + 'v2/images', text='{"images": []}')

    @mock.patch('glanceclient.v1.client.Client')
    def test_auth_plugin_invocation_with_v1(self, v1_client):
        args = '--os-image-api-version 1 image-list'
        glance_shell = openstack_shell.OpenStackImagesShell()
        glance_shell.main(args.split())
        self.assertEqual(1, self.v3_auth.call_count)
        self._assert_auth_plugin_args()

    @mock.patch('glanceclient.v2.client.Client')
    def test_auth_plugin_invocation_with_v2(self, v2_client):
        args = '--os-image-api-version 2 image-list'
        glance_shell = openstack_shell.OpenStackImagesShell()
        glance_shell.main(args.split())
        self.assertEqual(1, self.v3_auth.call_count)
        self._assert_auth_plugin_args()

    @mock.patch('glanceclient.v2.client.Client')
    def test_endpoint_from_interface(self, v2_client):
        args = '--os-image-api-version 2 image-list'
        glance_shell = openstack_shell.OpenStackImagesShell()
        glance_shell.main(args.split())
        assert v2_client.called
        args, kwargs = v2_client.call_args
        self.assertEqual(self.image_url, kwargs['endpoint_override'])

    def test_endpoint_real_from_interface(self):
        args = '--os-image-api-version 2 image-list'
        glance_shell = openstack_shell.OpenStackImagesShell()
        glance_shell.main(args.split())
        self.assertEqual(self.requests.request_history[2].url, self.image_url + f'v2/images?limit={DEFAULT_PAGE_SIZE}&sort_key=name&sort_dir=asc')