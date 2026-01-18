from __future__ import (absolute_import, division, print_function)
from collections.abc import Mapping, MutableMapping
from enum import Enum
from itertools import chain
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.utils.display import Display
from ansible.utils.vars import combine_vars
def _check_children_depth(self):
    depth = self.depth
    start_depth = self.depth
    seen = set([])
    unprocessed = set(self.child_groups)
    while unprocessed:
        seen.update(unprocessed)
        depth += 1
        to_process = unprocessed.copy()
        unprocessed = set([])
        for g in to_process:
            if g.depth < depth:
                g.depth = depth
                unprocessed.update(g.child_groups)
        if depth - start_depth > len(seen):
            raise AnsibleError("The group named '%s' has a recursive dependency loop." % to_native(self.name))