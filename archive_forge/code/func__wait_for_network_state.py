from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.dimensiondata import HAS_LIBCLOUD, DimensionDataModule
from ansible.module_utils.common.text.converters import to_native
def _wait_for_network_state(self, net_id, state_to_wait_for):
    try:
        return self.driver.connection.wait_for_state(state_to_wait_for, self.driver.ex_get_network_domain, self.module.params['wait_poll_interval'], self.module.params['wait_time'], net_id)
    except DimensionDataAPIException as e:
        self.module.fail_json(msg='Network did not reach % state in time: %s' % (state_to_wait_for, to_native(e)), exception=traceback.format_exc())