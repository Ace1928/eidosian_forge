import logging
import uuid
from osc_lib import exceptions
from osc_lib import utils as oscutils
from unittest import mock
from manilaclient import api_versions
from manilaclient.osc.v2 import (
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestUnsetShareGroupSnapshot(TestShareGroupSnapshot):

    def setUp(self):
        super(TestUnsetShareGroupSnapshot, self).setUp()
        self.share_group_snapshot = manila_fakes.FakeShareGroupSnapshot.create_one_share_group_snapshot()
        self.group_snapshot_mocks.get.return_value = self.share_group_snapshot
        self.cmd = osc_share_group_snapshots.UnsetShareGroupSnapshot(self.app, None)

    def test_unset_share_group_snapshot_name_description(self):
        arglist = [self.share_group_snapshot.id, '--name', '--description']
        verifylist = [('share_group_snapshot', self.share_group_snapshot.id), ('name', True), ('description', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.group_snapshot_mocks.update.assert_called_with(self.share_group_snapshot, name='', description='')
        self.assertIsNone(result)

    def test_unset_share_group_snapshot_name_exception(self):
        arglist = [self.share_group_snapshot.id, '--name']
        verifylist = [('share_group_snapshot', self.share_group_snapshot.id), ('name', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.group_snapshot_mocks.update.side_effect = Exception()
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)