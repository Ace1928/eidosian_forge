from __future__ import absolute_import, division, print_function
import os
import re
import tempfile
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems, string_types
def _current_cask_is_installed(self):
    if not self.valid_cask(self.current_cask):
        self.failed = True
        self.message = 'Invalid cask: {0}.'.format(self.current_cask)
        raise HomebrewCaskException(self.message)
    if self._brew_cask_command_is_deprecated():
        base_opts = [self.brew_path, 'list', '--cask']
    else:
        base_opts = [self.brew_path, 'cask', 'list']
    cmd = base_opts + [self.current_cask]
    rc, out, err = self.module.run_command(cmd)
    if rc == 0:
        return True
    else:
        return False