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
class TestAbandonShare(TestShare):

    def setUp(self):
        super(TestAbandonShare, self).setUp()
        self._share = manila_fakes.FakeShare.create_one_share(attrs={'status': 'available'})
        self.shares_mock.get.return_value = self._share
        self.cmd = osc_shares.AbandonShare(self.app, None)

    def test_share_abandon(self):
        arglist = [self._share.id]
        verifylist = [('share', [self._share.id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.shares_mock.unmanage.assert_called_with(self._share)
        self.assertIsNone(result)

    def test_share_abandon_wait(self):
        arglist = [self._share.id, '--wait']
        verifylist = [('share', [self._share.id]), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch('osc_lib.utils.wait_for_delete', return_value=True):
            result = self.cmd.take_action(parsed_args)
            self.shares_mock.unmanage.assert_called_with(self._share)
            self.assertIsNone(result)

    def test_share_abandon_wait_error(self):
        arglist = [self._share.id, '--wait']
        verifylist = [('share', [self._share.id]), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch('osc_lib.utils.wait_for_delete', return_value=False):
            self.assertRaises(osc_exceptions.CommandError, self.cmd.take_action, parsed_args)