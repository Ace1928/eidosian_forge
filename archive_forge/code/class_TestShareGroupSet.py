import argparse
from unittest import mock
import uuid
from osc_lib import exceptions
from osc_lib import exceptions as osc_exceptions
from osc_lib import utils as oscutils
from manilaclient.osc import utils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_groups as osc_share_groups
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareGroupSet(TestShareGroup):

    def setUp(self):
        super(TestShareGroupSet, self).setUp()
        self.share_group = manila_fakes.FakeShareGroup.create_one_share_group()
        self.share_group = manila_fakes.FakeShare.create_one_share(methods={'reset_state': None})
        self.groups_mock.get.return_value = self.share_group
        self.cmd = osc_share_groups.SetShareGroup(self.app, None)

    def test_set_share_group_name(self):
        new_name = uuid.uuid4().hex
        arglist = ['--name', new_name, self.share_group.id]
        verifylist = [('name', new_name), ('share_group', self.share_group.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.groups_mock.update.assert_called_with(self.share_group.id, name=parsed_args.name)

    def test_set_share_group_description(self):
        new_description = uuid.uuid4().hex
        arglist = ['--description', new_description, self.share_group.id]
        verifylist = [('description', new_description), ('share_group', self.share_group.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.groups_mock.update.assert_called_with(self.share_group.id, description=parsed_args.description)

    def test_share_group_set_status(self):
        new_status = 'available'
        arglist = [self.share_group.id, '--status', new_status]
        verifylist = [('share_group', self.share_group.id), ('status', new_status)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.share_group.reset_state.assert_called_with(new_status)
        self.assertIsNone(result)

    def test_share_group_set_status_exception(self):
        new_status = 'available'
        arglist = [self.share_group.id, '--status', new_status]
        verifylist = [('share_group', self.share_group.id), ('status', new_status)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.share_group.reset_state.side_effect = Exception()
        self.assertRaises(osc_exceptions.CommandError, self.cmd.take_action, parsed_args)