from __future__ import (absolute_import, division, print_function)
import itertools
import re
from ansible.module_utils.common._collections_compat import MutableMapping
from ansible.errors import AnsibleError
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import string_types
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.utils.display import Display
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def _populate_pool_groups(self, added_hosts):
    """Generate groups from Proxmox resource pools, ignoring VMs and
        containers that were skipped."""
    for pool in self._get_pools():
        poolid = pool.get('poolid')
        if not poolid:
            continue
        pool_group = self._group('pool_' + poolid)
        self.inventory.add_group(pool_group)
        for member in self._get_members_per_pool(poolid):
            name = member.get('name')
            if name and name in added_hosts:
                self.inventory.add_child(pool_group, name)