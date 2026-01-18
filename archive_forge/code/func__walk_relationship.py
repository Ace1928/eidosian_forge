from __future__ import (absolute_import, division, print_function)
from collections.abc import Mapping, MutableMapping
from enum import Enum
from itertools import chain
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.utils.display import Display
from ansible.utils.vars import combine_vars
def _walk_relationship(self, rel, include_self=False, preserve_ordering=False):
    """
        Given `rel` that is an iterable property of Group,
        consitituting a directed acyclic graph among all groups,
        Returns a set of all groups in full tree
        A   B    C
        |  / |  /
        | /  | /
        D -> E
        |  /    vertical connections
        | /     are directed upward
        F
        Called on F, returns set of (A, B, C, D, E)
        """
    seen = set([])
    unprocessed = set(getattr(self, rel))
    if include_self:
        unprocessed.add(self)
    if preserve_ordering:
        ordered = [self] if include_self else []
        ordered.extend(getattr(self, rel))
    while unprocessed:
        seen.update(unprocessed)
        new_unprocessed = set([])
        for new_item in chain.from_iterable((getattr(g, rel) for g in unprocessed)):
            new_unprocessed.add(new_item)
            if preserve_ordering:
                if new_item not in seen:
                    ordered.append(new_item)
        new_unprocessed.difference_update(seen)
        unprocessed = new_unprocessed
    if preserve_ordering:
        return ordered
    return seen