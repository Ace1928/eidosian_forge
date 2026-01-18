from __future__ import (absolute_import, division, print_function)
import fnmatch
import os
import sys
import re
import itertools
import traceback
from operator import attrgetter
from random import shuffle
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleOptionsError, AnsibleParserError
from ansible.inventory.data import InventoryData
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.parsing.utils.addresses import parse_address
from ansible.plugins.loader import inventory_loader
from ansible.utils.helpers import deduplicate_list
from ansible.utils.path import unfrackpath
from ansible.utils.display import Display
from ansible.utils.vars import combine_vars
from ansible.vars.plugins import get_vars_from_inventory_sources
def add_dynamic_host(self, host_info, result_item):
    """
        Helper function to add a new host to inventory based on a task result.
        """
    changed = False
    if not result_item.get('refresh'):
        self._cached_dynamic_hosts.append(host_info)
    if host_info:
        host_name = host_info.get('host_name')
        if host_name not in self.hosts:
            self.add_host(host_name, 'all')
            changed = True
        new_host = self.hosts.get(host_name)
        new_host_vars = new_host.get_vars()
        new_host_combined_vars = combine_vars(new_host_vars, host_info.get('host_vars', dict()))
        if new_host_vars != new_host_combined_vars:
            new_host.vars = new_host_combined_vars
            changed = True
        new_groups = host_info.get('groups', [])
        for group_name in new_groups:
            if group_name not in self.groups:
                group_name = self._inventory.add_group(group_name)
                changed = True
            new_group = self.groups[group_name]
            if new_group.add_host(self.hosts[host_name]):
                changed = True
        if changed:
            self.reconcile_inventory()
        result_item['changed'] = changed