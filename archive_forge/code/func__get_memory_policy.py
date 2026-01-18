from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def _get_memory_policy(self):
    memory_policy = self.param('memory_policy')
    if memory_policy == 'desktop':
        return 200
    elif memory_policy == 'server':
        return 150
    elif memory_policy == 'disabled':
        return 100