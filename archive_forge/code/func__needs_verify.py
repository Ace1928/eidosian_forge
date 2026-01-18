from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
def _needs_verify(self):
    cmd = self._get_cmd('verify')
    self._run_cmd(cmd)
    if self.rc != 0:
        self.failed = True
        self.msg = 'Failed to check for filesystem inconsistencies.'
    if self.FILES_NOT_MATCH in self.stdout:
        return True
    return False