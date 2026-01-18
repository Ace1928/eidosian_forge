import collections
import copy
import getpass
import json
import tempfile
from unittest import mock
from unittest.mock import call
import iso8601
from novaclient import api_versions
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils as common_utils
from openstackclient.compute.v2 import server
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
class TestEvacuateServer(TestServer):

    def setUp(self):
        super(TestEvacuateServer, self).setUp()
        self.image = image_fakes.create_one_image()
        self.image_client.get_image.return_value = self.image
        attrs = {'image': {'id': self.image.id}, 'networks': {}, 'adminPass': 'passw0rd'}
        new_server = compute_fakes.create_one_server(attrs=attrs)
        attrs['id'] = new_server.id
        methods = {'evacuate': new_server}
        self.server = compute_fakes.create_one_server(attrs=attrs, methods=methods)
        self.servers_mock.get.return_value = self.server
        self.cmd = server.EvacuateServer(self.app, None)

    def _test_evacuate(self, args, verify_args, evac_args):
        parsed_args = self.check_parser(self.cmd, args, verify_args)
        self.cmd.take_action(parsed_args)
        self.servers_mock.get.assert_called_with(self.server.id)
        self.server.evacuate.assert_called_with(**evac_args)

    def test_evacuate(self):
        args = [self.server.id]
        verify_args = [('server', self.server.id)]
        evac_args = {'host': None, 'on_shared_storage': False, 'password': None}
        self._test_evacuate(args, verify_args, evac_args)

    def test_evacuate_with_password(self):
        args = [self.server.id, '--password', 'password']
        verify_args = [('server', self.server.id), ('password', 'password')]
        evac_args = {'host': None, 'on_shared_storage': False, 'password': 'password'}
        self._test_evacuate(args, verify_args, evac_args)

    def test_evacuate_with_host(self):
        self.compute_client.api_version = api_versions.APIVersion('2.29')
        host = 'target-host'
        args = [self.server.id, '--host', 'target-host']
        verify_args = [('server', self.server.id), ('host', 'target-host')]
        evac_args = {'host': host, 'password': None}
        self._test_evacuate(args, verify_args, evac_args)

    def test_evacuate_with_host_pre_v229(self):
        self.compute_client.api_version = api_versions.APIVersion('2.28')
        args = [self.server.id, '--host', 'target-host']
        verify_args = [('server', self.server.id), ('host', 'target-host')]
        parsed_args = self.check_parser(self.cmd, args, verify_args)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_evacuate_without_share_storage(self):
        self.compute_client.api_version = api_versions.APIVersion('2.13')
        args = [self.server.id, '--shared-storage']
        verify_args = [('server', self.server.id), ('shared_storage', True)]
        evac_args = {'host': None, 'on_shared_storage': True, 'password': None}
        self._test_evacuate(args, verify_args, evac_args)

    def test_evacuate_without_share_storage_post_v213(self):
        self.compute_client.api_version = api_versions.APIVersion('2.14')
        args = [self.server.id, '--shared-storage']
        verify_args = [('server', self.server.id), ('shared_storage', True)]
        parsed_args = self.check_parser(self.cmd, args, verify_args)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    @mock.patch.object(common_utils, 'wait_for_status', return_value=True)
    def test_evacuate_with_wait_ok(self, mock_wait_for_status):
        args = [self.server.id, '--wait']
        verify_args = [('server', self.server.id), ('wait', True)]
        evac_args = {'host': None, 'on_shared_storage': False, 'password': None}
        self._test_evacuate(args, verify_args, evac_args)
        mock_wait_for_status.assert_called_once_with(self.servers_mock.get, self.server.id, callback=mock.ANY)