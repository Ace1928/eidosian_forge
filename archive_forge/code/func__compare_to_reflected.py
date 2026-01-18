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
def _compare_to_reflected(self, other: _constraint_sig[_C]) -> ComparisonResult:
    assert self._is_metadata
    metadata_obj = self
    conn_obj = other
    assert is_index_sig(conn_obj)
    return self.impl.compare_indexes(metadata_obj.const, conn_obj.const)