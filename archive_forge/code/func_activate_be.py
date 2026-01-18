from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
def activate_be(self):
    cmd = [self.module.get_bin_path('beadm'), 'activate', self.name]
    return self.module.run_command(cmd)