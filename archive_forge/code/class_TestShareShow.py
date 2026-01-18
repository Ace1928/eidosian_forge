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
class TestShareShow(TestShare):

    def setUp(self):
        super(TestShareShow, self).setUp()
        self._share = manila_fakes.FakeShare.create_one_share()
        self.shares_mock.get.return_value = self._share
        self._export_location = manila_fakes.FakeShareExportLocation.create_one_export_location()
        self.cmd = osc_shares.ShowShare(self.app, None)

    def test_share_show(self):
        arglist = [self._share.id]
        verifylist = [('share', self._share.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        cliutils.convert_dict_list_to_string = mock.Mock()
        cliutils.convert_dict_list_to_string.return_value = dict(self._export_location)
        columns, data = self.cmd.take_action(parsed_args)
        self.shares_mock.get.assert_called_with(self._share.id)
        self.assertEqual(manila_fakes.FakeShare.get_share_columns(self._share), columns)
        self.assertEqual(manila_fakes.FakeShare.get_share_data(self._share), data)