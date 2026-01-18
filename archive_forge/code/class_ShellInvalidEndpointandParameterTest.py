import argparse
import io
import json
import os
from unittest import mock
import subprocess
import tempfile
import testtools
from glanceclient import exc
from glanceclient import shell
import glanceclient.v1.client as client
import glanceclient.v1.images
import glanceclient.v1.shell as v1shell
from glanceclient.tests import utils
class ShellInvalidEndpointandParameterTest(utils.TestCase):

    def setUp(self):
        """Run before each test."""
        super(ShellInvalidEndpointandParameterTest, self).setUp()
        self.old_environment = os.environ.copy()
        os.environ = {'OS_USERNAME': 'username', 'OS_PASSWORD': 'password', 'OS_TENANT_ID': 'tenant_id', 'OS_TOKEN_ID': 'test', 'OS_AUTH_URL': 'http://127.0.0.1:5000/v2.0/', 'OS_AUTH_TOKEN': 'pass', 'OS_IMAGE_API_VERSION': '1', 'OS_REGION_NAME': 'test', 'OS_IMAGE_URL': 'http://is.invalid'}
        self.shell = shell.OpenStackImagesShell()
        self.patched = mock.patch('glanceclient.common.utils.get_data_file', autospec=True, return_value=None)
        self.mock_get_data_file = self.patched.start()
        self.gc = self._mock_glance_client()

    def _make_args(self, args):

        class Args(object):

            def __init__(self, entries):
                self.__dict__.update(entries)
        return Args(args)

    def _mock_glance_client(self):
        my_mocked_gc = mock.Mock()
        my_mocked_gc.get.return_value = {}
        return my_mocked_gc

    def tearDown(self):
        super(ShellInvalidEndpointandParameterTest, self).tearDown()
        os.environ = self.old_environment
        self.patched.stop()

    def run_command(self, cmd):
        self.shell.main(cmd.split())

    def assert_called(self, method, url, body=None, **kwargs):
        return self.shell.cs.assert_called(method, url, body, **kwargs)

    def assert_called_anytime(self, method, url, body=None):
        return self.shell.cs.assert_called_anytime(method, url, body)

    def test_image_list_invalid_endpoint(self):
        self.assertRaises(exc.CommunicationError, self.run_command, 'image-list')

    def test_image_create_invalid_endpoint(self):
        self.assertRaises(exc.CommunicationError, self.run_command, 'image-create')

    def test_image_delete_invalid_endpoint(self):
        self.assertRaises(exc.CommunicationError, self.run_command, 'image-delete <fake>')

    def test_image_download_invalid_endpoint(self):
        self.assertRaises(exc.CommunicationError, self.run_command, 'image-download <fake>')

    def test_members_list_invalid_endpoint(self):
        self.assertRaises(exc.CommunicationError, self.run_command, 'member-list --image-id fake')

    def test_image_show_invalid_endpoint(self):
        self.assertRaises(exc.CommunicationError, self.run_command, 'image-show --human-readable <IMAGE_ID>')

    def test_member_create_invalid_endpoint(self):
        self.assertRaises(exc.CommunicationError, self.run_command, 'member-create --can-share <IMAGE_ID> <TENANT_ID>')

    def test_member_delete_invalid_endpoint(self):
        self.assertRaises(exc.CommunicationError, self.run_command, 'member-delete  <IMAGE_ID> <TENANT_ID>')

    @mock.patch('sys.stderr')
    def test_image_create_invalid_size_parameter(self, __):
        self.assertRaises(SystemExit, self.run_command, 'image-create --size 10gb')

    @mock.patch('sys.stderr')
    def test_image_create_invalid_ram_parameter(self, __):
        self.assertRaises(SystemExit, self.run_command, 'image-create --min-ram 10gb')

    @mock.patch('sys.stderr')
    def test_image_create_invalid_min_disk_parameter(self, __):
        self.assertRaises(SystemExit, self.run_command, 'image-create --min-disk 10gb')

    @mock.patch('sys.stderr')
    def test_image_create_missing_disk_format(self, __):
        for origin in ('--file', '--location', '--copy-from'):
            e = self.assertRaises(exc.CommandError, self.run_command, '--os-image-api-version 1 image-create ' + origin + ' fake_src --container-format bare')
            self.assertEqual('error: Must provide --disk-format when using ' + origin + '.', e.message)

    @mock.patch('sys.stderr')
    def test_image_create_missing_container_format(self, __):
        for origin in ('--file', '--location', '--copy-from'):
            e = self.assertRaises(exc.CommandError, self.run_command, '--os-image-api-version 1 image-create ' + origin + ' fake_src --disk-format qcow2')
            self.assertEqual('error: Must provide --container-format when using ' + origin + '.', e.message)

    @mock.patch('sys.stderr')
    def test_image_create_missing_container_format_stdin_data(self, __):
        self.mock_get_data_file.return_value = io.StringIO()
        e = self.assertRaises(exc.CommandError, self.run_command, '--os-image-api-version 1 image-create --disk-format qcow2')
        self.assertEqual('error: Must provide --container-format when using stdin.', e.message)

    @mock.patch('sys.stderr')
    def test_image_create_missing_disk_format_stdin_data(self, __):
        self.mock_get_data_file.return_value = io.StringIO()
        e = self.assertRaises(exc.CommandError, self.run_command, '--os-image-api-version 1 image-create --container-format bare')
        self.assertEqual('error: Must provide --disk-format when using stdin.', e.message)

    @mock.patch('sys.stderr')
    def test_image_update_invalid_size_parameter(self, __):
        self.assertRaises(SystemExit, self.run_command, 'image-update --size 10gb')

    @mock.patch('sys.stderr')
    def test_image_update_invalid_min_disk_parameter(self, __):
        self.assertRaises(SystemExit, self.run_command, 'image-update --min-disk 10gb')

    @mock.patch('sys.stderr')
    def test_image_update_invalid_ram_parameter(self, __):
        self.assertRaises(SystemExit, self.run_command, 'image-update --min-ram 10gb')

    @mock.patch('sys.stderr')
    def test_image_list_invalid_min_size_parameter(self, __):
        self.assertRaises(SystemExit, self.run_command, 'image-list --size-min 10gb')

    @mock.patch('sys.stderr')
    def test_image_list_invalid_max_size_parameter(self, __):
        self.assertRaises(SystemExit, self.run_command, 'image-list --size-max 10gb')

    def test_do_image_list_with_changes_since(self):
        input = {'name': None, 'limit': None, 'status': None, 'container_format': 'bare', 'size_min': None, 'size_max': None, 'is_public': True, 'disk_format': 'raw', 'page_size': 20, 'visibility': True, 'member_status': 'Fake', 'owner': 'test', 'checksum': 'fake_checksum', 'tag': 'fake tag', 'properties': [], 'sort_key': None, 'sort_dir': None, 'all_tenants': False, 'human_readable': True, 'changes_since': '2011-1-1'}
        args = self._make_args(input)
        with mock.patch.object(self.gc.images, 'list') as mocked_list:
            mocked_list.return_value = {}
            v1shell.do_image_list(self.gc, args)
            exp_img_filters = {'container_format': 'bare', 'changes-since': '2011-1-1', 'disk_format': 'raw', 'is_public': True}
            mocked_list.assert_called_once_with(sort_dir=None, sort_key=None, owner='test', page_size=20, filters=exp_img_filters)