from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import ndp_proxy
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestListNDPProxy(TestNDPProxy):

    def setUp(self):
        super(TestListNDPProxy, self).setUp()
        attrs = {'router_id': self.router.id, 'port_id': self.port.id}
        ndp_proxies = network_fakes.create_ndp_proxies(attrs, count=3)
        self.columns = ('ID', 'Name', 'Router ID', 'IP Address', 'Project')
        self.data = []
        for np in ndp_proxies:
            self.data.append((np.id, np.name, np.router_id, np.ip_address, np.project_id))
        self.network_client.ndp_proxies = mock.Mock(return_value=ndp_proxies)
        self.cmd = ndp_proxy.ListNDPProxy(self.app, self.namespace)

    def test_ndp_proxy_list(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.ndp_proxies.assert_called_once_with()
        self.assertEqual(self.columns, columns)
        list_data = list(data)
        self.assertEqual(len(self.data), len(list_data))
        for index in range(len(list_data)):
            self.assertEqual(self.data[index], list_data[index])

    def test_ndp_proxy_list_router(self):
        arglist = ['--router', 'fake-router-name']
        verifylist = [('router', 'fake-router-name')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.ndp_proxies.assert_called_once_with(**{'router_id': 'fake-router-id'})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_ndp_proxy_list_port(self):
        arglist = ['--port', self.port.id]
        verifylist = [('port', self.port.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.ndp_proxies.assert_called_once_with(**{'port_id': self.port.id})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_ndp_proxy_list_name(self):
        arglist = ['--name', 'fake-ndp-proxy-name']
        verifylist = [('name', 'fake-ndp-proxy-name')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.ndp_proxies.assert_called_once_with(**{'name': 'fake-ndp-proxy-name'})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_ndp_proxy_list_ip_address(self):
        arglist = ['--ip-address', '2001::1:2']
        verifylist = [('ip_address', '2001::1:2')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.ndp_proxies.assert_called_once_with(**{'ip_address': '2001::1:2'})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_ndp_proxy_list_project(self):
        project = identity_fakes_v3.FakeProject.create_one_project()
        self.projects_mock.get.return_value = project
        arglist = ['--project', project.id]
        verifylist = [('project', project.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.ndp_proxies.assert_called_once_with(**{'project_id': project.id})
        self.assertEqual(self.columns, columns)
        self.assertItemsEqual(self.data, list(data))

    def test_ndp_proxy_list_project_domain(self):
        project = identity_fakes_v3.FakeProject.create_one_project()
        self.projects_mock.get.return_value = project
        arglist = ['--project', project.id, '--project-domain', project.domain_id]
        verifylist = [('project', project.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        filters = {'project_id': project.id}
        self.network_client.ndp_proxies.assert_called_once_with(**filters)
        self.assertEqual(self.columns, columns)
        self.assertItemsEqual(self.data, list(data))