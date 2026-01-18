import datetime
from unittest import mock
from openstackclient.compute.v2 import usage as usage_cmds
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestUsageShow(TestUsage):
    project = identity_fakes.FakeProject.create_one_project()
    usage = compute_fakes.create_one_usage(attrs={'project_id': project.name})
    columns = ('Project', 'Servers', 'RAM MB-Hours', 'CPU Hours', 'Disk GB-Hours')
    data = (usage_cmds.ProjectColumn(usage.project_id), usage_cmds.CountColumn(usage.server_usages), usage_cmds.FloatColumn(usage.total_memory_mb_usage), usage_cmds.FloatColumn(usage.total_vcpus_usage), usage_cmds.FloatColumn(usage.total_local_gb_usage))

    def setUp(self):
        super(TestUsageShow, self).setUp()
        self.compute_sdk_client.get_usage.return_value = self.usage
        self.projects_mock.get.return_value = self.project
        self.cmd = usage_cmds.ShowUsage(self.app, None)

    def test_usage_show_no_options(self):
        self.app.client_manager.auth_ref = mock.Mock()
        self.app.client_manager.auth_ref.project_id = self.project.id
        arglist = []
        verifylist = [('project', None), ('start', None), ('end', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_usage_show_with_options(self):
        arglist = ['--project', self.project.id, '--start', '2016-11-11', '--end', '2016-12-20']
        verifylist = [('project', self.project.id), ('start', '2016-11-11'), ('end', '2016-12-20')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.get_usage.assert_called_with(project=self.project.id, start=datetime.datetime(2016, 11, 11, 0, 0), end=datetime.datetime(2016, 12, 20, 0, 0))
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)