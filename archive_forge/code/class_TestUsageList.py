import datetime
from unittest import mock
from openstackclient.compute.v2 import usage as usage_cmds
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestUsageList(TestUsage):
    project = identity_fakes.FakeProject.create_one_project()
    usages = compute_fakes.create_usages(attrs={'project_id': project.name}, count=1)
    columns = ('Project', 'Servers', 'RAM MB-Hours', 'CPU Hours', 'Disk GB-Hours')
    data = [(usage_cmds.ProjectColumn(usages[0].project_id), usage_cmds.CountColumn(usages[0].server_usages), usage_cmds.FloatColumn(usages[0].total_memory_mb_usage), usage_cmds.FloatColumn(usages[0].total_vcpus_usage), usage_cmds.FloatColumn(usages[0].total_local_gb_usage))]

    def setUp(self):
        super(TestUsageList, self).setUp()
        self.compute_sdk_client.usages.return_value = self.usages
        self.projects_mock.list.return_value = [self.project]
        self.cmd = usage_cmds.ListUsage(self.app, None)

    def test_usage_list_no_options(self):
        arglist = []
        verifylist = [('start', None), ('end', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.projects_mock.list.assert_called_with()
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(tuple(self.data), tuple(data))

    def test_usage_list_with_options(self):
        arglist = ['--start', '2016-11-11', '--end', '2016-12-20']
        verifylist = [('start', '2016-11-11'), ('end', '2016-12-20')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.projects_mock.list.assert_called_with()
        self.compute_sdk_client.usages.assert_called_with(start=datetime.datetime(2016, 11, 11, 0, 0), end=datetime.datetime(2016, 12, 20, 0, 0), detailed=True)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(tuple(self.data), tuple(data))

    def test_usage_list_with_pagination(self):
        arglist = []
        verifylist = [('start', None), ('end', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.projects_mock.list.assert_called_with()
        self.compute_sdk_client.usages.assert_has_calls([mock.call(start=mock.ANY, end=mock.ANY, detailed=True)])
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(tuple(self.data), tuple(data))