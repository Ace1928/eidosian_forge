from __future__ import annotations
import difflib
import functools
import itertools
import textwrap
from collections import OrderedDict, defaultdict, deque
from typing import Any, Callable, Iterable, Mapping, overload
from optree import _C
from optree.registry import (
from optree.typing import (
from optree.typing import structseq as PyStructSequence  # noqa: N812
from optree.typing import structseq_fields
def _child_keys(tree: PyTree[T], is_leaf: Callable[[T], bool] | None=None, *, none_is_leaf: bool=False, namespace: str='') -> list[KeyPathEntry]:
    treespec = tree_structure(tree, is_leaf=is_leaf, none_is_leaf=none_is_leaf, namespace=namespace)
    assert not treespec_is_strict_leaf(treespec), 'treespec must be a non-leaf node'
    handler = register_keypaths.get(type(tree))
    if handler:
        return list(handler(tree))
    if is_structseq_instance(tree):
        return list(map(AttributeKeyPathEntry, structseq_fields(tree)))
    if is_namedtuple_instance(tree):
        return list(map(AttributeKeyPathEntry, namedtuple_fields(tree)))
    num_children = treespec.num_children
    return list(map(FlattenedKeyPathEntry, range(num_children)))