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
class TestShareGroupCreate(TestShareGroup):

    def setUp(self):
        super(TestShareGroupCreate, self).setUp()
        self.share_group = manila_fakes.FakeShareGroup.create_one_share_group()
        self.formatted_result = manila_fakes.FakeShareGroup.create_one_share_group(attrs={'id': self.share_group.id, 'created_at': self.share_group.created_at, 'project_id': self.share_group.project_id, 'share_group_type_id': self.share_group.share_group_type_id, 'share_types': '\n'.join(self.share_group.share_types)})
        self.groups_mock.create.return_value = self.share_group
        self.groups_mock.get.return_value = self.share_group
        self.cmd = osc_share_groups.CreateShareGroup(self.app, None)
        self.data = tuple(self.formatted_result._info.values())
        self.columns = tuple(self.share_group._info.keys())

    def test_share_group_create_no_args(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.groups_mock.create.assert_called_with(name=None, description=None, share_types=[], share_group_type=None, share_network=None, source_share_group_snapshot=None, availability_zone=None)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_share_group_create_with_options(self):
        arglist = ['--name', self.share_group.name, '--description', self.share_group.description]
        verifylist = [('name', self.share_group.name), ('description', self.share_group.description)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.groups_mock.create.assert_called_with(name=self.share_group.name, description=self.share_group.description, share_types=[], share_group_type=None, share_network=None, source_share_group_snapshot=None, availability_zone=None)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_share_group_create_az(self):
        arglist = ['--availability-zone', self.share_group.availability_zone]
        verifylist = [('availability_zone', self.share_group.availability_zone)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.groups_mock.create.assert_called_with(name=None, description=None, share_types=[], share_group_type=None, share_network=None, source_share_group_snapshot=None, availability_zone=self.share_group.availability_zone)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_share_group_create_share_types(self):
        share_types = manila_fakes.FakeShareType.create_share_types(count=2)
        self.share_types_mock.get = manila_fakes.FakeShareType.get_share_types(share_types)
        arglist = ['--share-types', share_types[0].id, share_types[1].id]
        verifylist = [('share_types', [share_types[0].id, share_types[1].id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.groups_mock.create.assert_called_with(name=None, description=None, share_types=share_types, share_group_type=None, share_network=None, source_share_group_snapshot=None, availability_zone=None)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_share_group_create_wait(self):
        arglist = ['--wait']
        verifylist = [('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.groups_mock.create.assert_called_with(name=None, description=None, share_types=[], share_group_type=None, share_network=None, source_share_group_snapshot=None, availability_zone=None)
        self.groups_mock.get.assert_called_with(self.share_group.id)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)