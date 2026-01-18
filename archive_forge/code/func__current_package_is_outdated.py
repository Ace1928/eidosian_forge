from __future__ import absolute_import, division, print_function
import json
import os.path
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems, string_types
def _current_package_is_outdated(self):
    if not self.valid_package(self.current_package):
        return False
    rc, out, err = self.module.run_command([self.brew_path, 'outdated', self.current_package])
    return rc != 0