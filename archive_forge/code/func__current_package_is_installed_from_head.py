from __future__ import absolute_import, division, print_function
import json
import os.path
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems, string_types
def _current_package_is_installed_from_head(self):
    if not Homebrew.valid_package(self.current_package):
        return False
    elif not self._current_package_is_installed():
        return False
    rc, out, err = self.module.run_command([self.brew_path, 'info', self.current_package])
    try:
        version_info = [line for line in out.split('\n') if line][0]
    except IndexError:
        return False
    return version_info.split(' ')[-1] == 'HEAD'