import ast
import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as tempest_lib_exc
import time
from manilaclient import config
from manilaclient import exceptions
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
def _get_expected_update_data(self, net_data, net_creation_data):
    default_return_value = None if utils.share_network_subnets_are_supported() else 'None'
    expected_nn_id = default_return_value if net_data.get('neutron_net_id') else net_creation_data.get('neutron_net_id', default_return_value)
    expected_nsn_id = default_return_value if net_data.get('neutron_subnet_id') else net_creation_data.get('neutron_subnet_id', default_return_value)
    return (expected_nn_id, expected_nsn_id)