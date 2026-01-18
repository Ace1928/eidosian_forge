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
class TestShareRevert(TestShare):

    def setUp(self):
        super(TestShareRevert, self).setUp()
        self.share = manila_fakes.FakeShare.create_one_share(attrs={'revert_to_snapshot_support': True}, methods={'revert_to_snapshot': None})
        self.share_snapshot = manila_fakes.FakeShareSnapshot.create_one_snapshot(attrs={'share_id': self.share.id})
        self.shares_mock.get.return_value = self.share
        self.snapshots_mock.get.return_value = self.share_snapshot
        self.cmd = osc_shares.RevertShare(self.app, None)

    def test_share_revert(self):
        arglist = [self.share_snapshot.id]
        verifylist = [('snapshot', self.share_snapshot.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.shares_mock.get.assert_called_with(self.share_snapshot.share_id)
        self.share.revert_to_snapshot.assert_called_with(self.share_snapshot)
        self.assertIsNone(result)

    def test_share_revert_exception(self):
        arglist = [self.share_snapshot.id]
        verifylist = [('snapshot', self.share_snapshot.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.share.revert_to_snapshot.side_effect = Exception()
        self.assertRaises(osc_exceptions.CommandError, self.cmd.take_action, parsed_args)