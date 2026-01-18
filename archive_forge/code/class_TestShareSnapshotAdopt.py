from unittest import mock
import uuid
import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common import cliutils
from manilaclient.osc.v2 import share_snapshots as osc_share_snapshots
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareSnapshotAdopt(TestShareSnapshot):

    def setUp(self):
        super(TestShareSnapshotAdopt, self).setUp()
        self.share = manila_fakes.FakeShare.create_one_share()
        self.shares_mock.get.return_value = self.share
        self.share_snapshot = manila_fakes.FakeShareSnapshot.create_one_snapshot(attrs={'status': 'available'})
        self.snapshots_mock.get.return_value = self.share_snapshot
        self.export_location = manila_fakes.FakeShareExportLocation.create_one_export_location()
        self.snapshots_mock.manage.return_value = self.share_snapshot
        self.cmd = osc_share_snapshots.AdoptShareSnapshot(self.app, None)
        self.data = tuple(self.share_snapshot._info.values())
        self.columns = tuple(self.share_snapshot._info.keys())

    def test_share_snapshot_adopt_missing_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_snapshot_adopt(self):
        arglist = [self.share.id, self.export_location.fake_path]
        verifylist = [('share', self.share.id), ('provider_location', self.export_location.fake_path)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.snapshots_mock.manage.assert_called_with(share=self.share, provider_location=self.export_location.fake_path, driver_options={}, name=None, description=None)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_snapshot_adopt_name(self):
        name = 'name-' + uuid.uuid4().hex
        arglist = [self.share.id, self.export_location.fake_path, '--name', name]
        verifylist = [('share', self.share.id), ('provider_location', self.export_location.fake_path), ('name', name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.snapshots_mock.manage.assert_called_with(share=self.share, provider_location=self.export_location.fake_path, driver_options={}, name=name, description=None)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_snapshot_adopt_driver_option(self):
        arglist = [self.share.id, self.export_location.fake_path, '--driver-option', 'key1=value1', '--driver-option', 'key2=value2']
        verifylist = [('share', self.share.id), ('provider_location', self.export_location.fake_path), ('driver_option', {'key1': 'value1', 'key2': 'value2'})]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.snapshots_mock.manage.assert_called_with(share=self.share, provider_location=self.export_location.fake_path, driver_options={'key1': 'value1', 'key2': 'value2'}, name=None, description=None)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_snapshot_adopt_wait(self):
        arglist = [self.share.id, self.export_location.fake_path, '--wait']
        verifylist = [('share', self.share.id), ('provider_location', self.export_location.fake_path), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.snapshots_mock.get.assert_called_with(self.share_snapshot.id)
        self.snapshots_mock.manage.assert_called_with(share=self.share, provider_location=self.export_location.fake_path, driver_options={}, name=None, description=None)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_snapshot_adopt_wait_error(self):
        arglist = [self.share.id, self.export_location.fake_path, '--wait']
        verifylist = [('share', self.share.id), ('provider_location', self.export_location.fake_path), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch('osc_lib.utils.wait_for_status', return_value=False):
            columns, data = self.cmd.take_action(parsed_args)
            self.snapshots_mock.get.assert_called_with(self.share_snapshot.id)
            self.snapshots_mock.manage.assert_called_with(share=self.share, provider_location=self.export_location.fake_path, driver_options={}, name=None, description=None)
            self.assertCountEqual(self.columns, columns)
            self.assertCountEqual(self.data, data)