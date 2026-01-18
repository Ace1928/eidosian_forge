from __future__ import absolute_import, division, print_function
import os
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def action_delete(module, existing_snapshots):
    commands = list()
    exist = False
    for snapshot in existing_snapshots:
        if module.params['snapshot_name'] == snapshot['name']:
            exist = True
    if exist:
        commands.append('snapshot delete {0}'.format(module.params['snapshot_name']))
    return commands