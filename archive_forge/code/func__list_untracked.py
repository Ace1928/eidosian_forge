from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def _list_untracked(self):
    args = ['purge', '--config', 'extensions.purge=', '-R', self.dest, '--print']
    return self._command(args)