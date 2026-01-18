from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
def _check_valid_policy(blade, policy):
    try:
        return bool(blade.get_object_store_access_policies(names=[policy]))
    except AttributeError:
        return False