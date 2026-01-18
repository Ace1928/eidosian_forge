from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def _get_storage_format(self):
    if self.param('storage_format') is not None:
        for sd_format in otypes.StorageFormat:
            if self.param('storage_format').lower() == str(sd_format):
                return sd_format