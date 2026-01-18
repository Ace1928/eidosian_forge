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
class TestShare(manila_fakes.TestShare):

    def setUp(self):
        super(TestShare, self).setUp()
        self.shares_mock = self.app.client_manager.share.shares
        self.shares_mock.reset_mock()
        self.export_locations_mock = self.app.client_manager.share.share_export_locations
        self.export_locations_mock.reset_mock()
        self.projects_mock = self.app.client_manager.identity.projects
        self.projects_mock.reset_mock()
        self.users_mock = self.app.client_manager.identity.users
        self.users_mock.reset_mock()
        self.snapshots_mock = self.app.client_manager.share.share_snapshots
        self.snapshots_mock.reset_mock()
        self.share_types_mock = self.app.client_manager.share.share_types
        self.share_types_mock.reset_mock()
        self.share_networks_mock = self.app.client_manager.share.share_networks
        self.share_networks_mock.reset_mock()
        self.app.client_manager.share.api_version = api_versions.APIVersion(MAX_VERSION)

    def setup_shares_mock(self, count):
        shares = manila_fakes.FakeShare.create_shares(count=count)
        self.shares_mock.get = manila_fakes.FakeShare.get_shares(shares, 0)
        return shares

    def setup_share_groups_mock(self):
        self.share_group_mock = self.app.client_manager.share.share_groups
        self.share_group_mock.reset_mock()
        share_group = manila_fakes.FakeShareGroup.create_one_share_group()
        self.share_group_mock.get = mock.Mock(return_value=share_group)
        return share_group