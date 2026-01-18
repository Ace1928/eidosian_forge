from __future__ import annotations
from typing import Any
from typing import ClassVar
from typing import Dict
from typing import Generic
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from sqlalchemy.sql.schema import Constraint
from sqlalchemy.sql.schema import ForeignKeyConstraint
from sqlalchemy.sql.schema import Index
from sqlalchemy.sql.schema import UniqueConstraint
from typing_extensions import TypeGuard
from .. import util
from ..util import sqla_compat
class _ix_constraint_sig(_constraint_sig[Index]):
    _is_index = True
    name: sqla_compat._ConstraintName

    @classmethod
    def _register(cls) -> None:
        _clsreg['index'] = cls

    def __init__(self, is_metadata: bool, impl: DefaultImpl, const: Index) -> None:
        self.impl = impl
        self.const = const
        self.name = const.name
        self.is_unique = bool(const.unique)
        self._is_metadata = is_metadata

    def _compare_to_reflected(self, other: _constraint_sig[_C]) -> ComparisonResult:
        assert self._is_metadata
        metadata_obj = self
        conn_obj = other
        assert is_index_sig(conn_obj)
        return self.impl.compare_indexes(metadata_obj.const, conn_obj.const)

    @util.memoized_property
    def has_expressions(self):
        return sqla_compat.is_expression_index(self.const)

    @util.memoized_property
    def column_names(self) -> Tuple[str, ...]:
        return tuple([col.name for col in self.const.columns])

    @util.memoized_property
    def column_names_optional(self) -> Tuple[Optional[str], ...]:
        return tuple([getattr(col, 'name', None) for col in self.const.expressions])

    @util.memoized_property
    def is_named(self):
        return True

    @util.memoized_property
    def unnamed(self):
        return (self.is_unique,) + self.column_names_optional