from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
def _get_pg(array, pod):
    """Return Protection Group or None"""
    try:
        return array.get_protection_groups(names=[pod])
    except Exception:
        return None