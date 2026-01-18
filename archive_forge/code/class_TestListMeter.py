from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network_meter
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestListMeter(TestMeter):
    meter_list = network_fakes.FakeNetworkMeter.create_meter(count=2)
    columns = ('ID', 'Name', 'Description', 'Shared')
    data = []
    for meters in meter_list:
        data.append((meters.id, meters.name, meters.description, meters.shared))

    def setUp(self):
        super(TestListMeter, self).setUp()
        self.network_client.metering_labels = mock.Mock(return_value=self.meter_list)
        self.cmd = network_meter.ListMeter(self.app, self.namespace)

    def test_meter_list(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.metering_labels.assert_called_with()
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))