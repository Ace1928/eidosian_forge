from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import router
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestListRouter(TestRouter):
    routers = network_fakes.FakeRouter.create_routers(count=3)
    extensions = network_fakes.create_one_extension()
    columns = ('ID', 'Name', 'Status', 'State', 'Project', 'Distributed', 'HA')
    columns_long = columns + ('Routes', 'External gateway info', 'Availability zones', 'Tags')
    columns_long_no_az = columns + ('Routes', 'External gateway info', 'Tags')
    data = []
    for r in routers:
        data.append((r.id, r.name, r.status, router.AdminStateColumn(r.admin_state_up), r.project_id, r.distributed, r.ha))
    router_agent_data = []
    for r in routers:
        router_agent_data.append((r.id, r.name, r.external_gateway_info))
    agents_columns = ('ID', 'Name', 'External Gateway Info')
    data_long = []
    for i in range(0, len(routers)):
        r = routers[i]
        data_long.append(data[i] + (router.RoutesColumn(r.routes), router.RouterInfoColumn(r.external_gateway_info), format_columns.ListColumn(r.availability_zones), format_columns.ListColumn(r.tags)))
    data_long_no_az = []
    for i in range(0, len(routers)):
        r = routers[i]
        data_long_no_az.append(data[i] + (router.RoutesColumn(r.routes), router.RouterInfoColumn(r.external_gateway_info), format_columns.ListColumn(r.tags)))

    def setUp(self):
        super(TestListRouter, self).setUp()
        self.cmd = router.ListRouter(self.app, self.namespace)
        self.network_client.agent_hosted_routers = mock.Mock(return_value=self.routers)
        self.network_client.routers = mock.Mock(return_value=self.routers)
        self.network_client.find_extension = mock.Mock(return_value=self.extensions)
        self.network_client.find_router = mock.Mock(return_value=self.routers[0])
        self._testagent = network_fakes.create_one_network_agent()
        self.network_client.get_agent = mock.Mock(return_value=self._testagent)
        self.network_client.get_router = mock.Mock(return_value=self.routers[0])

    def test_router_list_no_options(self):
        arglist = []
        verifylist = [('long', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.routers.assert_called_once_with()
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_router_list_no_ha_no_distributed(self):
        _routers = network_fakes.FakeRouter.create_routers({'ha': None, 'distributed': None}, count=3)
        arglist = []
        verifylist = [('long', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch.object(self.network_client, 'routers', return_value=_routers):
            columns, data = self.cmd.take_action(parsed_args)
        self.assertNotIn('is_distributed', columns)
        self.assertNotIn('is_ha', columns)

    def test_router_list_long(self):
        arglist = ['--long']
        verifylist = [('long', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.routers.assert_called_once_with()
        self.assertEqual(self.columns_long, columns)
        self.assertCountEqual(self.data_long, list(data))

    def test_router_list_long_no_az(self):
        arglist = ['--long']
        verifylist = [('long', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.network_client.find_extension = mock.Mock(return_value=None)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.routers.assert_called_once_with()
        self.assertEqual(self.columns_long_no_az, columns)
        self.assertCountEqual(self.data_long_no_az, list(data))

    def test_list_name(self):
        test_name = 'fakename'
        arglist = ['--name', test_name]
        verifylist = [('long', False), ('name', test_name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.routers.assert_called_once_with(**{'name': test_name})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_router_list_enable(self):
        arglist = ['--enable']
        verifylist = [('long', False), ('enable', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.routers.assert_called_once_with(**{'admin_state_up': True, 'is_admin_state_up': True})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_router_list_disable(self):
        arglist = ['--disable']
        verifylist = [('long', False), ('disable', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.routers.assert_called_once_with(**{'admin_state_up': False, 'is_admin_state_up': False})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_router_list_project(self):
        project = identity_fakes_v3.FakeProject.create_one_project()
        self.projects_mock.get.return_value = project
        arglist = ['--project', project.id]
        verifylist = [('project', project.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        filters = {'project_id': project.id}
        self.network_client.routers.assert_called_once_with(**filters)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_router_list_project_domain(self):
        project = identity_fakes_v3.FakeProject.create_one_project()
        self.projects_mock.get.return_value = project
        arglist = ['--project', project.id, '--project-domain', project.domain_id]
        verifylist = [('project', project.id), ('project_domain', project.domain_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        filters = {'project_id': project.id}
        self.network_client.routers.assert_called_once_with(**filters)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_router_list_agents_no_args(self):
        arglist = ['--agents']
        verifylist = []
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_router_list_agents(self):
        arglist = ['--agent', self._testagent.id]
        verifylist = [('agent', self._testagent.id)]
        attrs = {self._testagent.id}
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.agent_hosted_routers(*attrs)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_list_with_tag_options(self):
        arglist = ['--tags', 'red,blue', '--any-tags', 'red,green', '--not-tags', 'orange,yellow', '--not-any-tags', 'black,white']
        verifylist = [('tags', ['red', 'blue']), ('any_tags', ['red', 'green']), ('not_tags', ['orange', 'yellow']), ('not_any_tags', ['black', 'white'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.routers.assert_called_once_with(**{'tags': 'red,blue', 'any_tags': 'red,green', 'not_tags': 'orange,yellow', 'not_any_tags': 'black,white'})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))