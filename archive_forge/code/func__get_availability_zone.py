import ddt
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
def _get_availability_zone(self):
    availability_zones = self.user_client.list_availability_zones()
    return availability_zones[0]['Name']