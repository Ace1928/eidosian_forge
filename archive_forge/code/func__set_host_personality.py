from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def _set_host_personality(module, array):
    """Set host personality. Only called when supported"""
    if module.params['personality'] != 'delete':
        array.set_host(module.params['name'], personality=module.params['personality'])
    else:
        array.set_host(module.params['name'], personality='')