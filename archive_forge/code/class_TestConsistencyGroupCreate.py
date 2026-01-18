from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import consistency_group
class TestConsistencyGroupCreate(TestConsistencyGroup):
    volume_type = volume_fakes.create_one_volume_type()
    new_consistency_group = volume_fakes.create_one_consistency_group()
    consistency_group_snapshot = volume_fakes.create_one_consistency_group_snapshot()
    columns = ('availability_zone', 'created_at', 'description', 'id', 'name', 'status', 'volume_types')
    data = (new_consistency_group.availability_zone, new_consistency_group.created_at, new_consistency_group.description, new_consistency_group.id, new_consistency_group.name, new_consistency_group.status, new_consistency_group.volume_types)

    def setUp(self):
        super().setUp()
        self.consistencygroups_mock.create.return_value = self.new_consistency_group
        self.consistencygroups_mock.create_from_src.return_value = self.new_consistency_group
        self.consistencygroups_mock.get.return_value = self.new_consistency_group
        self.types_mock.get.return_value = self.volume_type
        self.cgsnapshots_mock.get.return_value = self.consistency_group_snapshot
        self.cmd = consistency_group.CreateConsistencyGroup(self.app, None)

    def test_consistency_group_create(self):
        arglist = ['--volume-type', self.volume_type.id, '--description', self.new_consistency_group.description, '--availability-zone', self.new_consistency_group.availability_zone, self.new_consistency_group.name]
        verifylist = [('volume_type', self.volume_type.id), ('description', self.new_consistency_group.description), ('availability_zone', self.new_consistency_group.availability_zone), ('name', self.new_consistency_group.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.types_mock.get.assert_called_once_with(self.volume_type.id)
        self.consistencygroups_mock.get.assert_not_called()
        self.consistencygroups_mock.create.assert_called_once_with(self.volume_type.id, name=self.new_consistency_group.name, description=self.new_consistency_group.description, availability_zone=self.new_consistency_group.availability_zone)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_consistency_group_create_without_name(self):
        arglist = ['--volume-type', self.volume_type.id, '--description', self.new_consistency_group.description, '--availability-zone', self.new_consistency_group.availability_zone]
        verifylist = [('volume_type', self.volume_type.id), ('description', self.new_consistency_group.description), ('availability_zone', self.new_consistency_group.availability_zone)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.types_mock.get.assert_called_once_with(self.volume_type.id)
        self.consistencygroups_mock.get.assert_not_called()
        self.consistencygroups_mock.create.assert_called_once_with(self.volume_type.id, name=None, description=self.new_consistency_group.description, availability_zone=self.new_consistency_group.availability_zone)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_consistency_group_create_from_source(self):
        arglist = ['--consistency-group-source', self.new_consistency_group.id, '--description', self.new_consistency_group.description, self.new_consistency_group.name]
        verifylist = [('source', self.new_consistency_group.id), ('description', self.new_consistency_group.description), ('name', self.new_consistency_group.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.types_mock.get.assert_not_called()
        self.consistencygroups_mock.get.assert_called_once_with(self.new_consistency_group.id)
        self.consistencygroups_mock.create_from_src.assert_called_with(None, self.new_consistency_group.id, name=self.new_consistency_group.name, description=self.new_consistency_group.description)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_consistency_group_create_from_snapshot(self):
        arglist = ['--consistency-group-snapshot', self.consistency_group_snapshot.id, '--description', self.new_consistency_group.description, self.new_consistency_group.name]
        verifylist = [('snapshot', self.consistency_group_snapshot.id), ('description', self.new_consistency_group.description), ('name', self.new_consistency_group.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.types_mock.get.assert_not_called()
        self.cgsnapshots_mock.get.assert_called_once_with(self.consistency_group_snapshot.id)
        self.consistencygroups_mock.create_from_src.assert_called_with(self.consistency_group_snapshot.id, None, name=self.new_consistency_group.name, description=self.new_consistency_group.description)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)