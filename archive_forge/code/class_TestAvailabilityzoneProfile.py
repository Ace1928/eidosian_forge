import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import availabilityzoneprofile
from octaviaclient.osc.v2 import constants
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
class TestAvailabilityzoneProfile(fakes.TestOctaviaClient):

    def setUp(self):
        super().setUp()
        self._availabilityzoneprofile = fakes.createFakeResource('availability_zone_profile')
        self.availabilityzoneprofile_info = copy.deepcopy(attr_consts.AVAILABILITY_ZONE_PROFILE_ATTRS)
        self.columns = copy.deepcopy(constants.AVAILABILITYZONEPROFILE_COLUMNS)
        self.api_mock = mock.Mock()
        mock_list = self.api_mock.availabilityzoneprofile_list
        mock_list.return_value = copy.deepcopy({'availability_zone_profiles': [attr_consts.AVAILABILITY_ZONE_PROFILE_ATTRS]})
        lb_client = self.app.client_manager
        lb_client.load_balancer = self.api_mock