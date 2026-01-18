from __future__ import annotations
from collections import deque
import collections.abc as collections_abc
import itertools
from itertools import zip_longest
import operator
import typing
from typing import Any
from typing import Callable
from typing import Deque
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from . import operators
from .cache_key import HasCacheKey
from .visitors import _TraverseInternalsType
from .visitors import anon_map
from .visitors import ExternallyTraversible
from .visitors import HasTraversalDispatch
from .visitors import HasTraverseInternals
from .. import util
from ..util import langhelpers
from ..util.typing import Self
class _GetChildrenTraversal(HasTraversalDispatch):
    """Generate a _children_traversal internal traversal dispatch for classes
    with a _traverse_internals collection."""

    def visit_has_cache_key(self, element, **kw):
        return ()

    def visit_clauseelement(self, element, **kw):
        return (element,)

    def visit_clauseelement_list(self, element, **kw):
        return element

    def visit_clauseelement_tuple(self, element, **kw):
        return element

    def visit_clauseelement_tuples(self, element, **kw):
        return itertools.chain.from_iterable(element)

    def visit_fromclause_canonical_column_collection(self, element, **kw):
        return ()

    def visit_string_clauseelement_dict(self, element, **kw):
        return element.values()

    def visit_fromclause_ordered_set(self, element, **kw):
        return element

    def visit_clauseelement_unordered_set(self, element, **kw):
        return element

    def visit_setup_join_tuple(self, element, **kw):
        for target, onclause, from_, flags in element:
            if from_ is not None:
                yield from_
            if not isinstance(target, str):
                yield _flatten_clauseelement(target)
            if onclause is not None and (not isinstance(onclause, str)):
                yield _flatten_clauseelement(onclause)

    def visit_memoized_select_entities(self, element, **kw):
        return self.visit_clauseelement_tuple(element, **kw)

    def visit_dml_ordered_values(self, element, **kw):
        for k, v in element:
            if hasattr(k, '__clause_element__'):
                yield k
            yield v

    def visit_dml_values(self, element, **kw):
        expr_values = {k for k in element if hasattr(k, '__clause_element__')}
        str_values = expr_values.symmetric_difference(element)
        for k in sorted(str_values):
            yield element[k]
        for k in expr_values:
            yield k
            yield element[k]

    def visit_dml_multi_values(self, element, **kw):
        return ()

    def visit_propagate_attrs(self, element, **kw):
        return ()