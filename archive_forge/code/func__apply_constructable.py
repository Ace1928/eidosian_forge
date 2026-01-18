from __future__ import (absolute_import, division, print_function)
import json
import ssl
from time import sleep
from ansible.errors import AnsibleError
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def _apply_constructable(self, name, variables):
    strict = self.get_option('strict')
    self._add_host_to_composed_groups(self.get_option('groups'), variables, name, strict=strict)
    self._add_host_to_keyed_groups(self.get_option('keyed_groups'), variables, name, strict=strict)
    self._set_composite_vars(self.get_option('compose'), variables, name, strict=strict)