from __future__ import (absolute_import, division, print_function)
from collections.abc import Mapping, MutableMapping
from enum import Enum
from itertools import chain
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.utils.display import Display
from ansible.utils.vars import combine_vars
def add_child_group(self, group):
    added = False
    if self == group:
        raise Exception("can't add group to itself")
    if group not in self.child_groups:
        start_ancestors = group.get_ancestors()
        new_ancestors = self.get_ancestors()
        if group in new_ancestors:
            raise AnsibleError("Adding group '%s' as child to '%s' creates a recursive dependency loop." % (to_native(group.name), to_native(self.name)))
        new_ancestors.add(self)
        new_ancestors.difference_update(start_ancestors)
        added = True
        self.child_groups.append(group)
        group.depth = max([self.depth + 1, group.depth])
        group._check_children_depth()
        if self.name not in [g.name for g in group.parent_groups]:
            group.parent_groups.append(self)
            for h in group.get_hosts():
                h.populate_ancestors(additions=new_ancestors)
        self.clear_hosts_cache()
    return added