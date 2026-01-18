from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network_meter
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestCreateMeter(TestMeter):
    project = identity_fakes_v3.FakeProject.create_one_project()
    domain = identity_fakes_v3.FakeDomain.create_one_domain()
    new_meter = network_fakes.FakeNetworkMeter.create_one_meter()
    columns = ('description', 'id', 'name', 'project_id', 'shared')
    data = (new_meter.description, new_meter.id, new_meter.name, new_meter.project_id, new_meter.shared)

    def setUp(self):
        super(TestCreateMeter, self).setUp()
        self.network_client.create_metering_label = mock.Mock(return_value=self.new_meter)
        self.projects_mock.get.return_value = self.project
        self.cmd = network_meter.CreateMeter(self.app, self.namespace)

    def test_create_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_create_default_options(self):
        arglist = [self.new_meter.name]
        verifylist = [('name', self.new_meter.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_metering_label.assert_called_once_with(**{'name': self.new_meter.name})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_create_all_options(self):
        arglist = ['--description', self.new_meter.description, '--project', self.new_meter.project_id, '--project-domain', self.domain.name, '--share', self.new_meter.name]
        verifylist = [('description', self.new_meter.description), ('name', self.new_meter.name), ('project', self.new_meter.project_id), ('project_domain', self.domain.name), ('share', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_metering_label.assert_called_once_with(**{'description': self.new_meter.description, 'name': self.new_meter.name, 'project_id': self.project.id, 'shared': True})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)