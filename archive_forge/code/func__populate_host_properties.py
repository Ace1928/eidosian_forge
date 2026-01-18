from __future__ import absolute_import, division, print_function
from ansible.errors import AnsibleError
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.six import text_type
from ansible_collections.community.vmware.plugins.plugin_utils.inventory import (
from ansible_collections.community.vmware.plugins.inventory.vmware_vm_inventory import BaseVMwareInventory
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
from ansible.parsing.yaml.objects import AnsibleVaultEncryptedUnicode
def _populate_host_properties(self, host_properties, host):
    self.inventory.add_host(host)
    strict = self.get_option('strict')
    compose = self.get_option('compose')
    self._set_composite_vars(compose, host_properties, host, strict=strict)
    self._add_host_to_composed_groups(self.get_option('groups'), host_properties, host, strict=strict)
    self._add_host_to_keyed_groups(self.get_option('keyed_groups'), host_properties, host, strict=strict)
    with_path = self.get_option('with_path')
    if with_path:
        parents = host_properties['path'].split('/')
        if parents:
            if isinstance(with_path, text_type):
                parents = [with_path] + parents
            c_name = self._sanitize_group_name('/'.join(parents))
            c_group = self.inventory.add_group(c_name)
            self.inventory.add_host(host, c_group)
            parents.pop()
            while len(parents) > 0:
                p_name = self._sanitize_group_name('/'.join(parents))
                p_group = self.inventory.add_group(p_name)
                self.inventory.add_child(p_group, c_group)
                c_group = p_group
                parents.pop()
    can_sanitize = self.get_option('with_sanitized_property_name')
    if can_sanitize:
        host_properties = camel_dict_to_snake_dict(host_properties)
    with_nested_properties = self.get_option('with_nested_properties')
    if with_nested_properties:
        for k, v in host_properties.items():
            k = self._sanitize_group_name(k) if can_sanitize else k
            self.inventory.set_variable(host, k, v)
    host_properties = to_flatten_dict(host_properties)
    for k, v in host_properties.items():
        k = self._sanitize_group_name(k) if can_sanitize else k
        self.inventory.set_variable(host, k, v)