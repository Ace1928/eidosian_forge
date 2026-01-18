import argparse
import ddt
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from osc_lib import exceptions as osc_exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.api_versions import MAX_VERSION
from manilaclient.common.apiclient import exceptions
from manilaclient.common import cliutils
from manilaclient.osc.v2 import share as osc_shares
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestAdoptShare(TestShare):

    def setUp(self):
        super(TestAdoptShare, self).setUp()
        self._share_type = manila_fakes.FakeShareType.create_one_sharetype()
        self.share_types_mock.get.return_value = self._share_type
        self._share = manila_fakes.FakeShare.create_one_share(attrs={'status': 'available', 'share_type': self._share_type.id, 'share_server_id': 'server-id' + uuid.uuid4().hex})
        self.shares_mock.get.return_value = self._share
        self.shares_mock.manage.return_value = self._share
        self.cmd = osc_shares.AdoptShare(self.app, None)
        self.datalist = tuple(self._share._info.values())
        self.columns = tuple(self._share._info.keys())

    def test_share_adopt_missing_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_share_adopt_required_args(self):
        arglist = ['some.host@driver#pool', 'NFS', '10.0.0.1:/example_path']
        verifylist = [('service_host', 'some.host@driver#pool'), ('protocol', 'NFS'), ('export_path', '10.0.0.1:/example_path')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.shares_mock.manage.assert_called_with(description=None, export_path='10.0.0.1:/example_path', name=None, protocol='NFS', service_host='some.host@driver#pool')
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.datalist, data)

    def test_share_adopt(self):
        arglist = ['some.host@driver#pool', 'NFS', '10.0.0.1:/example_path', '--name', self._share.id, '--description', self._share.description, '--share-type', self._share.share_type, '--driver-options', 'key1=value1', 'key2=value2', '--wait', '--public', '--share-server-id', self._share.share_server_id]
        verifylist = [('service_host', 'some.host@driver#pool'), ('protocol', 'NFS'), ('export_path', '10.0.0.1:/example_path'), ('name', self._share.id), ('description', self._share.description), ('share_type', self._share_type.id), ('driver_options', ['key1=value1', 'key2=value2']), ('wait', True), ('public', True), ('share_server_id', self._share.share_server_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.shares_mock.manage.assert_called_with(description=self._share.description, driver_options={'key1': 'value1', 'key2': 'value2'}, export_path='10.0.0.1:/example_path', name=self._share.id, protocol='NFS', service_host='some.host@driver#pool', share_server_id=self._share.share_server_id, share_type=self._share_type.id, public=True)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.datalist, data)

    @mock.patch('manilaclient.osc.v2.share.LOG')
    def test_share_adopt_wait_error(self, mock_logger):
        arglist = ['some.host@driver#pool', 'NFS', '10.0.0.1:/example_path', '--wait']
        verifylist = [('service_host', 'some.host@driver#pool'), ('protocol', 'NFS'), ('export_path', '10.0.0.1:/example_path'), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch('osc_lib.utils.wait_for_status', return_value=False):
            columns, data = self.cmd.take_action(parsed_args)
            self.shares_mock.manage.assert_called_with(description=None, export_path='10.0.0.1:/example_path', name=None, protocol='NFS', service_host='some.host@driver#pool')
            mock_logger.error.assert_called_with('ERROR: Share is in error state.')
            self.shares_mock.get.assert_called_with(self._share.id)
            self.assertCountEqual(self.columns, columns)
            self.assertCountEqual(self.datalist, data)

    def test_share_adopt_visibility_api_version_exception(self):
        self.app.client_manager.share.api_version = api_versions.APIVersion('2.7')
        arglist = ['some.host@driver#pool', 'NFS', '10.0.0.1:/example_path', '--public']
        verifylist = [('service_host', 'some.host@driver#pool'), ('protocol', 'NFS'), ('export_path', '10.0.0.1:/example_path'), ('public', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(osc_exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_share_adopt_share_server_api_version_exception(self):
        self.app.client_manager.share.api_version = api_versions.APIVersion('2.48')
        arglist = ['some.host@driver#pool', 'NFS', '10.0.0.1:/example_path', '--share-server-id', self._share.share_server_id]
        verifylist = [('service_host', 'some.host@driver#pool'), ('protocol', 'NFS'), ('export_path', '10.0.0.1:/example_path'), ('share_server_id', self._share.share_server_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(osc_exceptions.CommandError, self.cmd.take_action, parsed_args)