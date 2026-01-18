import ast
import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as tempest_lib_exc
import time
from manilaclient import config
from manilaclient import exceptions
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
def _wait_for_update_security_service_compatible_result(self, share_network, current_security_service, new_security_service=None):
    compatible_expected_result = 'True'
    check_is_compatible = 'None'
    tentatives = 0
    while check_is_compatible != compatible_expected_result:
        tentatives += 1
        if not new_security_service:
            check_is_compatible = self.user_client.share_network_security_service_add_check(share_network['id'], current_security_service['id'])['compatible']
        else:
            check_is_compatible = self.user_client.share_network_security_service_update_check(share_network['id'], current_security_service['id'], new_security_service['id'])['compatible']
        if tentatives > 3:
            timeout_message = "Share network security service add/update check did not reach 'compatible=True' within 15 seconds."
            raise exceptions.TimeoutException(message=timeout_message)
        time.sleep(5)