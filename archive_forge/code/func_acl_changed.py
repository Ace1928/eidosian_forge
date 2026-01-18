from __future__ import absolute_import, division, print_function
import os
import platform
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
def acl_changed(module, cmd):
    """Returns true if the provided command affects the existing ACLs, false otherwise."""
    if platform.system().lower() == 'freebsd':
        return True
    cmd = cmd[:]
    cmd.insert(1, '--test')
    lines = run_acl(module, cmd)
    for line in lines:
        if not line.endswith('*,*'):
            return True
    return False