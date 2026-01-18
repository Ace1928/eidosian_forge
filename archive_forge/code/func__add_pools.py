from __future__ import (absolute_import, division, print_function)
import json
import ssl
from time import sleep
from ansible.errors import AnsibleError
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def _add_pools(self, pools):
    for pool in pools.values():
        group_name = 'xo_pool_{0}'.format(clean_group_name(pool['name_label']))
        self.inventory.add_group(group_name)