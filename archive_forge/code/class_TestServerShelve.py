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
class TestServerShelve(TestServer):

    def setUp(self):
        super().setUp()
        self.server = compute_fakes.create_one_sdk_server(attrs={'status': 'ACTIVE'})
        self.compute_sdk_client.find_server.return_value = self.server
        self.compute_sdk_client.shelve_server.return_value = None
        self.cmd = server.ShelveServer(self.app, None)

    def test_shelve(self):
        arglist = [self.server.name]
        verifylist = [('servers', [self.server.name]), ('wait', False), ('offload', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertIsNone(result)
        self.compute_sdk_client.find_server.assert_called_with(self.server.name, ignore_missing=False)
        self.compute_sdk_client.shelve_server.assert_called_with(self.server.id)
        self.compute_sdk_client.shelve_offload_server.assert_not_called()

    def test_shelve_already_shelved(self):
        self.server.status = 'SHELVED'
        arglist = [self.server.name]
        verifylist = [('servers', [self.server.name]), ('wait', False), ('offload', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertIsNone(result)
        self.compute_sdk_client.find_server.assert_called_with(self.server.name, ignore_missing=False)
        self.compute_sdk_client.shelve_server.assert_not_called()
        self.compute_sdk_client.shelve_offload_server.assert_not_called()

    @mock.patch.object(common_utils, 'wait_for_status', return_value=True)
    def test_shelve_with_wait(self, mock_wait_for_status):
        arglist = ['--wait', self.server.name]
        verifylist = [('servers', [self.server.name]), ('wait', True), ('offload', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertIsNone(result)
        self.compute_sdk_client.find_server.assert_called_with(self.server.name, ignore_missing=False)
        self.compute_sdk_client.shelve_server.assert_called_with(self.server.id)
        self.compute_sdk_client.shelve_offload_server.assert_not_called()
        mock_wait_for_status.assert_called_once_with(self.compute_sdk_client.get_server, self.server.id, callback=mock.ANY, success_status=('shelved', 'shelved_offloaded'))

    @mock.patch.object(common_utils, 'wait_for_status', return_value=True)
    def test_shelve_offload(self, mock_wait_for_status):
        arglist = ['--offload', self.server.name]
        verifylist = [('servers', [self.server.name]), ('wait', False), ('offload', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertIsNone(result)
        self.compute_sdk_client.find_server.assert_called_once_with(self.server.name, ignore_missing=False)
        self.compute_sdk_client.get_server.assert_called_once_with(self.server.id)
        self.compute_sdk_client.shelve_server.assert_called_with(self.server.id)
        self.compute_sdk_client.shelve_offload_server.assert_called_once_with(self.server.id)
        mock_wait_for_status.assert_called_once_with(self.compute_sdk_client.get_server, self.server.id, callback=mock.ANY, success_status=('shelved', 'shelved_offloaded'))