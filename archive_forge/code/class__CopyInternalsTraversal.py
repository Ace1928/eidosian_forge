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
class _CopyInternalsTraversal(HasTraversalDispatch):
    """Generate a _copy_internals internal traversal dispatch for classes
    with a _traverse_internals collection."""

    def visit_clauseelement(self, attrname, parent, element, clone=_clone, **kw):
        return clone(element, **kw)

    def visit_clauseelement_list(self, attrname, parent, element, clone=_clone, **kw):
        return [clone(clause, **kw) for clause in element]

    def visit_clauseelement_tuple(self, attrname, parent, element, clone=_clone, **kw):
        return tuple([clone(clause, **kw) for clause in element])

    def visit_executable_options(self, attrname, parent, element, clone=_clone, **kw):
        return tuple([clone(clause, **kw) for clause in element])

    def visit_clauseelement_unordered_set(self, attrname, parent, element, clone=_clone, **kw):
        return {clone(clause, **kw) for clause in element}

    def visit_clauseelement_tuples(self, attrname, parent, element, clone=_clone, **kw):
        return [tuple((clone(tup_elem, **kw) for tup_elem in elem)) for elem in element]

    def visit_string_clauseelement_dict(self, attrname, parent, element, clone=_clone, **kw):
        return {key: clone(value, **kw) for key, value in element.items()}

    def visit_setup_join_tuple(self, attrname, parent, element, clone=_clone, **kw):
        return tuple(((clone(target, **kw) if target is not None else None, clone(onclause, **kw) if onclause is not None else None, clone(from_, **kw) if from_ is not None else None, flags) for target, onclause, from_, flags in element))

    def visit_memoized_select_entities(self, attrname, parent, element, **kw):
        return self.visit_clauseelement_tuple(attrname, parent, element, **kw)

    def visit_dml_ordered_values(self, attrname, parent, element, clone=_clone, **kw):
        return [(clone(key, **kw) if hasattr(key, '__clause_element__') else key, clone(value, **kw)) for key, value in element]

    def visit_dml_values(self, attrname, parent, element, clone=_clone, **kw):
        return {clone(key, **kw) if hasattr(key, '__clause_element__') else key: clone(value, **kw) for key, value in element.items()}

    def visit_dml_multi_values(self, attrname, parent, element, clone=_clone, **kw):

        def copy(elem):
            if isinstance(elem, (list, tuple)):
                return [clone(value, **kw) if hasattr(value, '__clause_element__') else value for value in elem]
            elif isinstance(elem, dict):
                return {clone(key, **kw) if hasattr(key, '__clause_element__') else key: clone(value, **kw) if hasattr(value, '__clause_element__') else value for key, value in elem.items()}
            else:
                assert False
        return [[copy(sub_element) for sub_element in sequence] for sequence in element]

    def visit_propagate_attrs(self, attrname, parent, element, clone=_clone, **kw):
        return element