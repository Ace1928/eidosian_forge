from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def _connect_new_volume(module, array, answer=False):
    """Connect volume to host"""
    api_version = array._list_available_rest_versions()
    if AC_REQUIRED_API_VERSION in api_version and module.params['lun']:
        answer = True
        if not module.check_mode:
            try:
                array.connect_host(module.params['name'], module.params['volume'], lun=module.params['lun'])
            except Exception:
                module.fail_json(msg='LUN ID {0} invalid. Check for duplicate LUN IDs.'.format(module.params['lun']))
    else:
        answer = True
        if not module.check_mode:
            array.connect_host(module.params['name'], module.params['volume'])
    return answer