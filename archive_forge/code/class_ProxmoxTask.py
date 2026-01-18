from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.proxmox import (
class ProxmoxTask:

    def __init__(self, task):
        self.info = dict()
        for k, v in task.items():
            if k == 'status' and isinstance(v, str):
                self.info[k] = v
                if v != 'OK':
                    self.info['failed'] = True
            else:
                self.info[k] = v