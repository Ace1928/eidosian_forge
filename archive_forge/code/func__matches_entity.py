from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def _matches_entity(self, item, entity):
    return equal(item.get('id'), entity.id) and equal(item.get('name'), entity.name)