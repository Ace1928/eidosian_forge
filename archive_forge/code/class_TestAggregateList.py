from unittest import mock
from unittest.mock import call
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.compute.v2 import aggregate
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
class TestAggregateList(TestAggregate):
    list_columns = ('ID', 'Name', 'Availability Zone')
    list_columns_long = ('ID', 'Name', 'Availability Zone', 'Properties', 'Hosts')
    list_data = ((TestAggregate.fake_ag.id, TestAggregate.fake_ag.name, TestAggregate.fake_ag.availability_zone),)
    list_data_long = ((TestAggregate.fake_ag.id, TestAggregate.fake_ag.name, TestAggregate.fake_ag.availability_zone, format_columns.DictColumn({key: value for key, value in TestAggregate.fake_ag.metadata.items() if key != 'availability_zone'}), format_columns.ListColumn(TestAggregate.fake_ag.hosts)),)

    def setUp(self):
        super(TestAggregateList, self).setUp()
        self.compute_sdk_client.aggregates.return_value = [self.fake_ag]
        self.cmd = aggregate.ListAggregate(self.app, None)

    def test_aggregate_list(self):
        parsed_args = self.check_parser(self.cmd, [], [])
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.list_columns, columns)
        self.assertCountEqual(self.list_data, tuple(data))

    def test_aggregate_list_with_long(self):
        arglist = ['--long']
        vertifylist = [('long', True)]
        parsed_args = self.check_parser(self.cmd, arglist, vertifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.list_columns_long, columns)
        self.assertCountEqual(self.list_data_long, tuple(data))