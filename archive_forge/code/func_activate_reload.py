from __future__ import absolute_import, division, print_function
import time
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def activate_reload(module, pkg, flag):
    iteration = 0
    if flag:
        cmd = 'install activate {0} forced'.format(pkg)
    else:
        cmd = 'install deactivate {0} forced'.format(pkg)
    opts = {'ignore_timeout': True}
    while iteration < 10:
        msg = load_config(module, [cmd], True, opts)
        if msg:
            if isinstance(msg[0], int):
                if msg[0] == -32603:
                    return cmd
            elif isinstance(msg[0], str):
                if 'socket is closed' in msg[0].lower():
                    return cmd
                if 'another install operation is in progress' in msg[0].lower() or 'failed' in msg[0].lower():
                    time.sleep(2)
        iteration += 1