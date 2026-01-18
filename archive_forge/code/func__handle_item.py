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
def _handle_item(self, node, ittype, item):
    """Handle an item from the list of LXC containers and Qemu VM. The
        return value will be either None if the item was skipped or the name of
        the item if it was added to the inventory."""
    if item.get('template'):
        return None
    properties = dict()
    name, vmid = (item['name'], item['vmid'])
    want_facts = self.get_option('want_facts')
    if want_facts:
        self._get_vm_status(properties, node, vmid, ittype, name)
        self._get_vm_config(properties, node, vmid, ittype, name)
        self._get_vm_snapshots(properties, node, vmid, ittype, name)
    if not self._can_add_host(name, properties):
        return None
    self._add_host(name, properties)
    node_type_group = self._group('%s_%s' % (node, ittype))
    self.inventory.add_child(self._group('all_' + ittype), name)
    self.inventory.add_child(node_type_group, name)
    item_status = item['status']
    if item_status == 'running':
        if want_facts and ittype == 'qemu' and self.get_option('qemu_extended_statuses'):
            item_status = properties.get(self._fact('qmpstatus'), item_status)
    self.inventory.add_child(self._group('all_%s' % (item_status,)), name)
    return name