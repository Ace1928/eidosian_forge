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
class TestShareSnapshotCreate(TestShareSnapshot):

    def setUp(self):
        super(TestShareSnapshotCreate, self).setUp()
        self.share = manila_fakes.FakeShare.create_one_share()
        self.shares_mock.create.return_value = self.share
        self.shares_mock.get.return_value = self.share
        self.share_snapshot = manila_fakes.FakeShareSnapshot.create_one_snapshot(attrs={'status': 'available'})
        self.snapshots_mock.get.return_value = self.share_snapshot
        self.snapshots_mock.create.return_value = self.share_snapshot
        self.cmd = osc_share_snapshots.CreateShareSnapshot(self.app, None)
        self.data = tuple(self.share_snapshot._info.values())
        self.columns = tuple(self.share_snapshot._info.keys())

    def test_share_snapshot_create_missing_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_share_snapshot_create_required_args(self):
        arglist = [self.share.id]
        verifylist = [('share', self.share.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.snapshots_mock.create.assert_called_with(share=self.share, force=False, name=None, description=None, metadata={})
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_share_snapshot_create_force(self):
        arglist = [self.share.id, '--force']
        verifylist = [('share', self.share.id), ('force', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.snapshots_mock.create.assert_called_with(share=self.share, force=True, name=None, description=None, metadata={})
        self.assertCountEqual(columns, columns)
        self.assertCountEqual(self.data, data)

    def test_share_snapshot_create(self):
        arglist = [self.share.id, '--name', self.share_snapshot.name, '--description', self.share_snapshot.description]
        verifylist = [('share', self.share.id), ('name', self.share_snapshot.name), ('description', self.share_snapshot.description)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.snapshots_mock.create.assert_called_with(share=self.share, force=False, name=self.share_snapshot.name, description=self.share_snapshot.description, metadata={})
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_share_snapshot_create_metadata(self):
        arglist = [self.share.id, '--name', self.share_snapshot.name, '--description', self.share_snapshot.description, '--property', 'Manila=zorilla', '--property', 'Zorilla=manila']
        verifylist = [('share', self.share.id), ('name', self.share_snapshot.name), ('description', self.share_snapshot.description), ('property', {'Manila': 'zorilla', 'Zorilla': 'manila'})]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.snapshots_mock.create.assert_called_with(share=self.share, force=False, name=self.share_snapshot.name, description=self.share_snapshot.description, metadata={'Manila': 'zorilla', 'Zorilla': 'manila'})
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_share_snapshot_create_wait(self):
        arglist = [self.share.id, '--wait']
        verifylist = [('share', self.share.id), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.snapshots_mock.create.assert_called_with(share=self.share, force=False, name=None, description=None, metadata={})
        self.snapshots_mock.get.assert_called_with(self.share_snapshot.id)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    @mock.patch('manilaclient.osc.v2.share_snapshots.LOG')
    def test_share_snapshot_create_wait_error(self, mock_logger):
        arglist = [self.share.id, '--wait']
        verifylist = [('share', self.share.id), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch('osc_lib.utils.wait_for_status', return_value=False):
            columns, data = self.cmd.take_action(parsed_args)
            self.snapshots_mock.create.assert_called_with(share=self.share, force=False, name=None, description=None, metadata={})
            mock_logger.error.assert_called_with('ERROR: Share snapshot is in error state.')
            self.snapshots_mock.get.assert_called_with(self.share_snapshot.id)
            self.assertCountEqual(self.columns, columns)
            self.assertCountEqual(self.data, data)