from __future__ import absolute_import, division, print_function
import os
import re
import tempfile
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems, string_types
def _upgrade_all(self):
    if self.module.check_mode:
        self.changed = True
        self.message = 'Casks would be upgraded.'
        raise HomebrewCaskException(self.message)
    if self._brew_cask_command_is_deprecated():
        cmd = [self.brew_path, 'upgrade', '--cask']
    else:
        cmd = [self.brew_path, 'cask', 'upgrade']
    if self.greedy:
        cmd = cmd + ['--greedy']
    rc, out, err = ('', '', '')
    if self.sudo_password:
        rc, out, err = self._run_command_with_sudo_password(cmd)
    else:
        rc, out, err = self.module.run_command(cmd)
    if rc == 0:
        if re.search('==> No Casks to upgrade', out.strip(), re.IGNORECASE):
            self.message = 'Homebrew casks already upgraded.'
        else:
            self.changed = True
            self.message = 'Homebrew casks upgraded.'
        return True
    else:
        self.failed = True
        self.message = err.strip()
        raise HomebrewCaskException(self.message)