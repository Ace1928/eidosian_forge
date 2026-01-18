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
class ShellCacheSchemaTest(testutils.TestCase):

    def setUp(self):
        super(ShellCacheSchemaTest, self).setUp()
        self._mock_client_setup()
        self._mock_shell_setup()
        self.cache_dir = '/dir_for_cached_schema'
        self.os_auth_url = 'http://localhost:5000/v2'
        url_hex = hashlib.sha1(self.os_auth_url.encode('utf-8')).hexdigest()
        self.prefix_path = self.cache_dir + '/' + url_hex
        self.cache_files = [self.prefix_path + '/image_schema.json', self.prefix_path + '/namespace_schema.json', self.prefix_path + '/resource_type_schema.json']

    def tearDown(self):
        super(ShellCacheSchemaTest, self).tearDown()

    def _mock_client_setup(self):
        self.schema_dict = {'name': 'image', 'properties': {'name': {'type': 'string', 'description': 'Name of image'}}}
        self.client = mock.Mock()
        schema_odict = OrderedDict(self.schema_dict)
        self.client.schemas.get.return_value = schemas.Schema(schema_odict)

    def _mock_shell_setup(self):
        self.shell = openstack_shell.OpenStackImagesShell()
        self.shell._get_versioned_client = mock.create_autospec(self.shell._get_versioned_client, return_value=self.client, spec_set=True)

    def _make_args(self, args):

        class Args(object):

            def __init__(self, entries):
                self.__dict__.update(entries)
        return Args(args)

    @mock.patch('builtins.open', new=mock.mock_open(), create=True)
    @mock.patch('os.path.exists', return_value=True)
    def test_cache_schemas_gets_when_forced(self, exists_mock):
        options = {'get_schema': True, 'os_auth_url': self.os_auth_url}
        schema_odict = OrderedDict(self.schema_dict)
        args = self._make_args(options)
        client = self.shell._get_versioned_client('2', args)
        self.shell._cache_schemas(args, client, home_dir=self.cache_dir)
        self.assertEqual(12, open.mock_calls.__len__())
        self.assertEqual(mock.call(self.cache_files[0], 'w'), open.mock_calls[0])
        self.assertEqual(mock.call(self.cache_files[1], 'w'), open.mock_calls[4])
        actual = json.loads(open.mock_calls[2][1][0])
        self.assertEqual(schema_odict, actual)
        actual = json.loads(open.mock_calls[6][1][0])
        self.assertEqual(schema_odict, actual)

    @mock.patch('builtins.open', new=mock.mock_open(), create=True)
    @mock.patch('os.path.exists', side_effect=[True, False, False, False])
    def test_cache_schemas_gets_when_not_exists(self, exists_mock):
        options = {'get_schema': False, 'os_auth_url': self.os_auth_url}
        schema_odict = OrderedDict(self.schema_dict)
        args = self._make_args(options)
        client = self.shell._get_versioned_client('2', args)
        self.shell._cache_schemas(args, client, home_dir=self.cache_dir)
        self.assertEqual(12, open.mock_calls.__len__())
        self.assertEqual(mock.call(self.cache_files[0], 'w'), open.mock_calls[0])
        self.assertEqual(mock.call(self.cache_files[1], 'w'), open.mock_calls[4])
        actual = json.loads(open.mock_calls[2][1][0])
        self.assertEqual(schema_odict, actual)
        actual = json.loads(open.mock_calls[6][1][0])
        self.assertEqual(schema_odict, actual)

    @mock.patch('builtins.open', new=mock.mock_open(), create=True)
    @mock.patch('os.path.exists', return_value=True)
    def test_cache_schemas_leaves_when_present_not_forced(self, exists_mock):
        options = {'get_schema': False, 'os_auth_url': self.os_auth_url}
        client = mock.MagicMock()
        self.shell._cache_schemas(self._make_args(options), client, home_dir=self.cache_dir)
        exists_mock.assert_has_calls([mock.call(self.prefix_path), mock.call(self.cache_files[0]), mock.call(self.cache_files[1]), mock.call(self.cache_files[2])])
        self.assertEqual(4, exists_mock.call_count)
        self.assertEqual(0, open.mock_calls.__len__())

    @mock.patch('builtins.open', new=mock.mock_open(), create=True)
    @mock.patch('os.path.exists', return_value=True)
    def test_cache_schemas_leaves_auto_switch(self, exists_mock):
        options = {'get_schema': True, 'os_auth_url': self.os_auth_url}
        self.client.schemas.get.return_value = Exception()
        client = mock.MagicMock()
        switch_version = self.shell._cache_schemas(self._make_args(options), client, home_dir=self.cache_dir)
        self.assertEqual(True, switch_version)