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
class HasShallowCopy(HasTraverseInternals):
    """attribute-wide operations that are useful for classes that use
    __slots__ and therefore can't operate on their attributes in a dictionary.


    """
    __slots__ = ()
    if typing.TYPE_CHECKING:

        def _generated_shallow_copy_traversal(self, other: Self) -> None:
            ...

        def _generated_shallow_from_dict_traversal(self, d: Dict[str, Any]) -> None:
            ...

        def _generated_shallow_to_dict_traversal(self) -> Dict[str, Any]:
            ...

    @classmethod
    def _generate_shallow_copy(cls, internal_dispatch: _TraverseInternalsType, method_name: str) -> Callable[[Self, Self], None]:
        code = '\n'.join((f'    other.{attrname} = self.{attrname}' for attrname, _ in internal_dispatch))
        meth_text = f'def {method_name}(self, other):\n{code}\n'
        return langhelpers._exec_code_in_env(meth_text, {}, method_name)

    @classmethod
    def _generate_shallow_to_dict(cls, internal_dispatch: _TraverseInternalsType, method_name: str) -> Callable[[Self], Dict[str, Any]]:
        code = ',\n'.join((f"    '{attrname}': self.{attrname}" for attrname, _ in internal_dispatch))
        meth_text = f'def {method_name}(self):\n    return {{{code}}}\n'
        return langhelpers._exec_code_in_env(meth_text, {}, method_name)

    @classmethod
    def _generate_shallow_from_dict(cls, internal_dispatch: _TraverseInternalsType, method_name: str) -> Callable[[Self, Dict[str, Any]], None]:
        code = '\n'.join((f"    self.{attrname} = d['{attrname}']" for attrname, _ in internal_dispatch))
        meth_text = f'def {method_name}(self, d):\n{code}\n'
        return langhelpers._exec_code_in_env(meth_text, {}, method_name)

    def _shallow_from_dict(self, d: Dict[str, Any]) -> None:
        cls = self.__class__
        shallow_from_dict: Callable[[HasShallowCopy, Dict[str, Any]], None]
        try:
            shallow_from_dict = cls.__dict__['_generated_shallow_from_dict_traversal']
        except KeyError:
            shallow_from_dict = self._generate_shallow_from_dict(cls._traverse_internals, '_generated_shallow_from_dict_traversal')
            cls._generated_shallow_from_dict_traversal = shallow_from_dict
        shallow_from_dict(self, d)

    def _shallow_to_dict(self) -> Dict[str, Any]:
        cls = self.__class__
        shallow_to_dict: Callable[[HasShallowCopy], Dict[str, Any]]
        try:
            shallow_to_dict = cls.__dict__['_generated_shallow_to_dict_traversal']
        except KeyError:
            shallow_to_dict = self._generate_shallow_to_dict(cls._traverse_internals, '_generated_shallow_to_dict_traversal')
            cls._generated_shallow_to_dict_traversal = shallow_to_dict
        return shallow_to_dict(self)

    def _shallow_copy_to(self, other: Self) -> None:
        cls = self.__class__
        shallow_copy: Callable[[Self, Self], None]
        try:
            shallow_copy = cls.__dict__['_generated_shallow_copy_traversal']
        except KeyError:
            shallow_copy = self._generate_shallow_copy(cls._traverse_internals, '_generated_shallow_copy_traversal')
            cls._generated_shallow_copy_traversal = shallow_copy
        shallow_copy(self, other)

    def _clone(self, **kw: Any) -> Self:
        """Create a shallow copy"""
        c = self.__class__.__new__(self.__class__)
        self._shallow_copy_to(c)
        return c