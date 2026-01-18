from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.lxd import (
from ansible.module_utils.basic import AnsibleModule
import os
def _needs_to_change_project_config(self, key):
    if key not in self.config:
        return False
    old_configs = self.old_project_json['metadata'].get(key, None)
    return self.config[key] != old_configs