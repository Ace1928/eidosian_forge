import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import availabilityzone
from octaviaclient.osc.v2 import constants
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
class TestAvailabilityzoneCreate(TestAvailabilityzone):

    def setUp(self):
        super().setUp()
        self.api_mock.availabilityzone_create.return_value = {'availability_zone': self.availabilityzone_info}
        lb_client = self.app.client_manager
        lb_client.load_balancer = self.api_mock
        self.cmd = availabilityzone.CreateAvailabilityzone(self.app, None)

    @mock.patch('octaviaclient.osc.v2.utils.get_availabilityzone_attrs')
    def test_availabilityzone_create(self, mock_client):
        mock_client.return_value = self.availabilityzone_info
        arglist = ['--name', self._availabilityzone.name, '--availabilityzoneprofile', 'mock_azpf_id', '--description', 'description for availabilityzone']
        verifylist = [('availabilityzoneprofile', 'mock_azpf_id'), ('name', self._availabilityzone.name), ('description', 'description for availabilityzone')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.availabilityzone_create.assert_called_with(json={'availability_zone': self.availabilityzone_info})