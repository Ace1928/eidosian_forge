from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def __get_minor(self, full_version):
    if full_version is None:
        return None
    if isinstance(full_version, otypes.Version):
        return full_version.minor
    return int(full_version.split('.')[1])