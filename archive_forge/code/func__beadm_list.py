from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
def _beadm_list(self):
    cmd = [self.module.get_bin_path('beadm'), 'list', '-H']
    if '@' in self.name:
        cmd.append('-s')
    return self.module.run_command(cmd)