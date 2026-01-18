from __future__ import absolute_import, division, print_function
import os
import re
import tempfile
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems, string_types
def _current_cask_is_outdated(self):
    if not self.valid_cask(self.current_cask):
        return False
    if self._brew_cask_command_is_deprecated():
        base_opts = [self.brew_path, 'outdated', '--cask']
    else:
        base_opts = [self.brew_path, 'cask', 'outdated']
    cask_is_outdated_command = base_opts + (['--greedy'] if self.greedy else []) + [self.current_cask]
    rc, out, err = self.module.run_command(cask_is_outdated_command)
    return out != ''