from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def _disconnect_volume(module, array, answer=False):
    """Disconnect volume from host"""
    answer = True
    if not module.check_mode:
        try:
            array.disconnect_host(module.params['name'], module.params['volume'])
        except Exception:
            module.fail_json(msg='Failed to disconnect volume {0}'.format(module.params['volume']))
    return answer