from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase
from ansible.utils.vars import merge_hash
def _get_async_dir(self):
    async_dir = self.get_shell_option('async_dir', default='~/.ansible_async')
    return self._remote_expand_user(async_dir)