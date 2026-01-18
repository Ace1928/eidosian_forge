from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleError
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
def _add_groups(self):
    """Add Linode instance groups to the dynamic inventory."""
    self.linode_groups = set(filter(None, [instance.group for instance in self.instances]))
    for linode_group in self.linode_groups:
        self.inventory.add_group(linode_group)