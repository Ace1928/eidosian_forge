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
class _fk_constraint_sig(_constraint_sig[ForeignKeyConstraint]):
    _is_fk = True

    @classmethod
    def _register(cls) -> None:
        _clsreg['foreign_key_constraint'] = cls

    def __init__(self, is_metadata: bool, impl: DefaultImpl, const: ForeignKeyConstraint) -> None:
        self._is_metadata = is_metadata
        self.impl = impl
        self.const = const
        self.name = sqla_compat.constraint_name_or_none(const.name)
        self.source_schema, self.source_table, self.source_columns, self.target_schema, self.target_table, self.target_columns, onupdate, ondelete, deferrable, initially = sqla_compat._fk_spec(const)
        self._sig: Tuple[Any, ...] = (self.source_schema, self.source_table, tuple(self.source_columns), self.target_schema, self.target_table, tuple(self.target_columns)) + ((None if onupdate.lower() == 'no action' else onupdate.lower()) if onupdate else None, (None if ondelete.lower() == 'no action' else ondelete.lower()) if ondelete else None, 'initially_deferrable' if initially and initially.lower() == 'deferred' else 'deferrable' if deferrable else 'not deferrable')

    @util.memoized_property
    def unnamed_no_options(self):
        return (self.source_schema, self.source_table, tuple(self.source_columns), self.target_schema, self.target_table, tuple(self.target_columns))