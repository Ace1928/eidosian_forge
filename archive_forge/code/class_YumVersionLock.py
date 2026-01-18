from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from fnmatch import fnmatch
class YumVersionLock:

    def __init__(self, module):
        self.module = module
        self.params = module.params
        self.yum_bin = module.get_bin_path('yum', required=True)

    def get_versionlock_packages(self):
        """ Get an overview of all packages on yum versionlock """
        rc, out, err = self.module.run_command([self.yum_bin, 'versionlock', 'list'])
        if rc == 0:
            return out
        elif rc == 1 and 'o such command:' in err:
            self.module.fail_json(msg='Error: Please install rpm package yum-plugin-versionlock : ' + to_native(err) + to_native(out))
        self.module.fail_json(msg='Error: ' + to_native(err) + to_native(out))

    def ensure_state(self, packages, command):
        """ Ensure packages state """
        rc, out, err = self.module.run_command([self.yum_bin, '-q', 'versionlock', command] + packages)
        if 'No package found for' in out:
            self.module.fail_json(msg=out)
        if rc == 0:
            return True
        self.module.fail_json(msg='Error: ' + to_native(err) + to_native(out))