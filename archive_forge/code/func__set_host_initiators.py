from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def _set_host_initiators(module, array):
    """Set host initiators."""
    if module.params['nqn']:
        try:
            array.set_host(module.params['name'], nqnlist=module.params['nqn'])
        except Exception:
            module.fail_json(msg='Setting of NVMe NQN failed.')
    if module.params['iqn']:
        try:
            array.set_host(module.params['name'], iqnlist=module.params['iqn'])
        except Exception:
            module.fail_json(msg='Setting of iSCSI IQN failed.')
    if module.params['wwns']:
        try:
            array.set_host(module.params['name'], wwnlist=module.params['wwns'])
        except Exception:
            module.fail_json(msg='Setting of FC WWNs failed.')