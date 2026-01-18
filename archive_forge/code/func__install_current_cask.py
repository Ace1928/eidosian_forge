from __future__ import absolute_import, division, print_function
import os
import re
import tempfile
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems, string_types
def _install_current_cask(self):
    if not self.valid_cask(self.current_cask):
        self.failed = True
        self.message = 'Invalid cask: {0}.'.format(self.current_cask)
        raise HomebrewCaskException(self.message)
    if '--force' not in self.install_options and self._current_cask_is_installed():
        self.unchanged_count += 1
        self.message = 'Cask already installed: {0}'.format(self.current_cask)
        return True
    if self.module.check_mode:
        self.changed = True
        self.message = 'Cask would be installed: {0}'.format(self.current_cask)
        raise HomebrewCaskException(self.message)
    if self._brew_cask_command_is_deprecated():
        base_opts = [self.brew_path, 'install', '--cask']
    else:
        base_opts = [self.brew_path, 'cask', 'install']
    opts = base_opts + [self.current_cask] + self.install_options
    cmd = [opt for opt in opts if opt]
    rc, out, err = ('', '', '')
    if self.sudo_password:
        rc, out, err = self._run_command_with_sudo_password(cmd)
    else:
        rc, out, err = self.module.run_command(cmd)
    if self._current_cask_is_installed():
        self.changed_count += 1
        self.changed = True
        self.message = 'Cask installed: {0}'.format(self.current_cask)
        return True
    elif self.accept_external_apps and re.search('Error: It seems there is already an App at', err):
        self.unchanged_count += 1
        self.message = 'Cask already installed: {0}'.format(self.current_cask)
        return True
    else:
        self.failed = True
        self.message = err.strip()
        raise HomebrewCaskException(self.message)