from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network_meter
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestShowMeter(TestMeter):
    new_meter = network_fakes.FakeNetworkMeter.create_one_meter()
    columns = ('description', 'id', 'name', 'project_id', 'shared')
    data = (new_meter.description, new_meter.id, new_meter.name, new_meter.project_id, new_meter.shared)

    def setUp(self):
        super(TestShowMeter, self).setUp()
        self.cmd = network_meter.ShowMeter(self.app, self.namespace)
        self.network_client.find_metering_label = mock.Mock(return_value=self.new_meter)

    def test_show_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_meter_show_option(self):
        arglist = [self.new_meter.name]
        verifylist = [('meter', self.new_meter.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.find_metering_label.assert_called_with(self.new_meter.name, ignore_missing=False)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)