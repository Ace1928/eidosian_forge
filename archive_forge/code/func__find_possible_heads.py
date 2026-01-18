import itertools
from .. import debug, revision, trace
from ..graph import DictParentsProvider, Graph, invert_parent_map
from ..repository import AbstractSearchResult
def _find_possible_heads(parent_map, tip_keys, depth):
    """Walk backwards (towards children) through the parent_map.

    This finds 'heads' that will hopefully succinctly describe our search
    graph.
    """
    child_map = invert_parent_map(parent_map)
    heads = set()
    current_roots = tip_keys
    walked = set(current_roots)
    while current_roots and depth > 0:
        depth -= 1
        children = set()
        children_update = children.update
        for p in current_roots:
            try:
                children_update(child_map[p])
            except KeyError:
                heads.add(p)
        children = children.difference(walked)
        walked.update(children)
        current_roots = children
    if current_roots:
        heads.update(current_roots)
    return heads