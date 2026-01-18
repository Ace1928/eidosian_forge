import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import availabilityzoneprofile
from octaviaclient.osc.v2 import constants
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
class TestAvailabilityzoneProfileCreate(TestAvailabilityzoneProfile):

    def setUp(self):
        super().setUp()
        self.api_mock.availabilityzoneprofile_create.return_value = {'availability_zone_profile': self.availabilityzoneprofile_info}
        lb_client = self.app.client_manager
        lb_client.load_balancer = self.api_mock
        self.cmd = availabilityzoneprofile.CreateAvailabilityzoneProfile(self.app, None)

    @mock.patch('octaviaclient.osc.v2.utils.get_availabilityzoneprofile_attrs')
    def test_availabilityzoneprofile_create(self, mock_client):
        mock_client.return_value = self.availabilityzoneprofile_info
        arglist = ['--name', self._availabilityzoneprofile.name, '--provider', 'mock_provider', '--availability-zone-data', '{"mock_key": "mock_value"}']
        verifylist = [('provider', 'mock_provider'), ('name', self._availabilityzoneprofile.name), ('availability_zone_data', '{"mock_key": "mock_value"}')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.availabilityzoneprofile_create.assert_called_with(json={'availability_zone_profile': self.availabilityzoneprofile_info})