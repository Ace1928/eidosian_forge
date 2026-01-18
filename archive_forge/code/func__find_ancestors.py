from io import BytesIO
from ..lazy_import import lazy_import
import bisect
import math
import tempfile
import zlib
from .. import (chunk_writer, debug, fifo_cache, lru_cache, osutils, trace,
from . import index, static_tuple
from .index import _OPTION_KEY_ELEMENTS, _OPTION_LEN, _OPTION_NODE_REFS
def _find_ancestors(self, keys, ref_list_num, parent_map, missing_keys):
    """Find the parent_map information for the set of keys.

        This populates the parent_map dict and missing_keys set based on the
        queried keys. It also can fill out an arbitrary number of parents that
        it finds while searching for the supplied keys.

        It is unlikely that you want to call this directly. See
        "CombinedGraphIndex.find_ancestry()" for a more appropriate API.

        :param keys: A keys whose ancestry we want to return
            Every key will either end up in 'parent_map' or 'missing_keys'.
        :param ref_list_num: This index in the ref_lists is the parents we
            care about.
        :param parent_map: {key: parent_keys} for keys that are present in this
            index. This may contain more entries than were in 'keys', that are
            reachable ancestors of the keys requested.
        :param missing_keys: keys which are known to be missing in this index.
            This may include parents that were not directly requested, but we
            were able to determine that they are not present in this index.
        :return: search_keys    parents that were found but not queried to know
            if they are missing or present. Callers can re-query this index for
            those keys, and they will be placed into parent_map or missing_keys
        """
    if not self.key_count():
        missing_keys.update(keys)
        return set()
    if ref_list_num >= self.node_ref_lists:
        raise ValueError('No ref list %d, index has %d ref lists' % (ref_list_num, self.node_ref_lists))
    nodes, nodes_and_keys = self._walk_through_internal_nodes(keys)
    parents_not_on_page = set()
    for node_index, sub_keys in nodes_and_keys:
        if not sub_keys:
            continue
        node = nodes[node_index]
        parents_to_check = set()
        for next_sub_key in sub_keys:
            if next_sub_key not in node:
                missing_keys.add(next_sub_key)
            else:
                value, refs = node[next_sub_key]
                parent_keys = refs[ref_list_num]
                parent_map[next_sub_key] = parent_keys
                parents_to_check.update(parent_keys)
        parents_to_check = parents_to_check.difference(parent_map)
        while parents_to_check:
            next_parents_to_check = set()
            for key in parents_to_check:
                if key in node:
                    value, refs = node[key]
                    parent_keys = refs[ref_list_num]
                    parent_map[key] = parent_keys
                    next_parents_to_check.update(parent_keys)
                elif key < node.min_key:
                    parents_not_on_page.add(key)
                elif key > node.max_key:
                    parents_not_on_page.add(key)
                else:
                    missing_keys.add(key)
            parents_to_check = next_parents_to_check.difference(parent_map)
    search_keys = parents_not_on_page.difference(parent_map).difference(missing_keys)
    return search_keys