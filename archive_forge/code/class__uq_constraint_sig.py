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
class _uq_constraint_sig(_constraint_sig[UniqueConstraint]):
    _is_uq = True

    @classmethod
    def _register(cls) -> None:
        _clsreg['unique_constraint'] = cls
    is_unique = True

    def __init__(self, is_metadata: bool, impl: DefaultImpl, const: UniqueConstraint) -> None:
        self.impl = impl
        self.const = const
        self.name = sqla_compat.constraint_name_or_none(const.name)
        self._sig = tuple(sorted([col.name for col in const.columns]))
        self._is_metadata = is_metadata

    @property
    def column_names(self) -> Tuple[str, ...]:
        return tuple([col.name for col in self.const.columns])

    def _compare_to_reflected(self, other: _constraint_sig[_C]) -> ComparisonResult:
        assert self._is_metadata
        metadata_obj = self
        conn_obj = other
        assert is_uq_sig(conn_obj)
        return self.impl.compare_unique_constraint(metadata_obj.const, conn_obj.const)