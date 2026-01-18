from __future__ import absolute_import, division, print_function
import json
import os.path
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems, string_types
def _current_package_is_installed(self):
    if not self.valid_package(self.current_package):
        self.failed = True
        self.message = 'Invalid package: {0}.'.format(self.current_package)
        raise HomebrewException(self.message)
    cmd = ['{brew_path}'.format(brew_path=self.brew_path), 'info', '--json=v2', self.current_package]
    rc, out, err = self.module.run_command(cmd)
    if err:
        self.failed = True
        self.message = err.strip()
        raise HomebrewException(self.message)
    data = json.loads(out)
    return _check_package_in_json(data, 'formulae') or _check_package_in_json(data, 'casks')