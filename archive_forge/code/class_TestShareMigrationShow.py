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
class TestShareMigrationShow(TestShare):

    def setUp(self):
        super(TestShareMigrationShow, self).setUp()
        self._share = manila_fakes.FakeShare.create_one_share(attrs={'status': 'available', 'task_state': 'migration_in_progress'}, methods={'migration_get_progress': ('<Response [200]>', {'total_progress': 0, 'task_state': 'migration_in_progress', 'details': {}})})
        self.shares_mock.get.return_value = self._share
        self.shares_mock.manage.return_value = self._share
        self.cmd = osc_shares.ShareMigrationShow(self.app, None)

    def test_migration_show(self):
        arglist = [self._share.id]
        verifylist = [('share', self._share.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self._share.migration_get_progress.assert_called