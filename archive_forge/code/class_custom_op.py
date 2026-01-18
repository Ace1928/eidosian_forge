from __future__ import annotations
from enum import IntEnum
from operator import add as _uncast_add
from operator import and_ as _uncast_and_
from operator import contains as _uncast_contains
from operator import eq as _uncast_eq
from operator import floordiv as _uncast_floordiv
from operator import ge as _uncast_ge
from operator import getitem as _uncast_getitem
from operator import gt as _uncast_gt
from operator import inv as _uncast_inv
from operator import le as _uncast_le
from operator import lshift as _uncast_lshift
from operator import lt as _uncast_lt
from operator import mod as _uncast_mod
from operator import mul as _uncast_mul
from operator import ne as _uncast_ne
from operator import neg as _uncast_neg
from operator import or_ as _uncast_or_
from operator import rshift as _uncast_rshift
from operator import sub as _uncast_sub
from operator import truediv as _uncast_truediv
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Optional
from typing import overload
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .. import exc
from .. import util
from ..util.typing import Literal
from ..util.typing import Protocol
class custom_op(OperatorType, Generic[_T]):
    """Represent a 'custom' operator.

    :class:`.custom_op` is normally instantiated when the
    :meth:`.Operators.op` or :meth:`.Operators.bool_op` methods
    are used to create a custom operator callable.  The class can also be
    used directly when programmatically constructing expressions.   E.g.
    to represent the "factorial" operation::

        from sqlalchemy.sql import UnaryExpression
        from sqlalchemy.sql import operators
        from sqlalchemy import Numeric

        unary = UnaryExpression(table.c.somecolumn,
                modifier=operators.custom_op("!"),
                type_=Numeric)


    .. seealso::

        :meth:`.Operators.op`

        :meth:`.Operators.bool_op`

    """
    __name__ = 'custom_op'
    __slots__ = ('opstring', 'precedence', 'is_comparison', 'natural_self_precedent', 'eager_grouping', 'return_type', 'python_impl')

    def __init__(self, opstring: str, precedence: int=0, is_comparison: bool=False, return_type: Optional[Union[Type[TypeEngine[_T]], TypeEngine[_T]]]=None, natural_self_precedent: bool=False, eager_grouping: bool=False, python_impl: Optional[Callable[..., Any]]=None):
        self.opstring = opstring
        self.precedence = precedence
        self.is_comparison = is_comparison
        self.natural_self_precedent = natural_self_precedent
        self.eager_grouping = eager_grouping
        self.return_type = return_type._to_instance(return_type) if return_type else None
        self.python_impl = python_impl

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, custom_op) and other._hash_key() == self._hash_key()

    def __hash__(self) -> int:
        return hash(self._hash_key())

    def _hash_key(self) -> Union[CacheConst, Tuple[Any, ...]]:
        return (self.__class__, self.opstring, self.precedence, self.is_comparison, self.natural_self_precedent, self.eager_grouping, self.return_type._static_cache_key if self.return_type else None)

    @overload
    def __call__(self, left: ColumnExpressionArgument[Any], right: Optional[Any]=None, *other: Any, **kwargs: Any) -> ColumnElement[Any]:
        ...

    @overload
    def __call__(self, left: Operators, right: Optional[Any]=None, *other: Any, **kwargs: Any) -> Operators:
        ...

    def __call__(self, left: Any, right: Optional[Any]=None, *other: Any, **kwargs: Any) -> Operators:
        if hasattr(left, '__sa_operate__'):
            return left.operate(self, right, *other, **kwargs)
        elif self.python_impl:
            return self.python_impl(left, right, *other, **kwargs)
        else:
            raise exc.InvalidRequestError(f"Custom operator {self.opstring!r} can't be used with plain Python objects unless it includes the 'python_impl' parameter.")