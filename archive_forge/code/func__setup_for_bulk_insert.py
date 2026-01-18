from __future__ import annotations
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import overload
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import attributes
from . import context
from . import evaluator
from . import exc as orm_exc
from . import loading
from . import persistence
from .base import NO_VALUE
from .context import AbstractORMCompileState
from .context import FromStatement
from .context import ORMFromStatementCompileState
from .context import QueryContext
from .. import exc as sa_exc
from .. import util
from ..engine import Dialect
from ..engine import result as _result
from ..sql import coercions
from ..sql import dml
from ..sql import expression
from ..sql import roles
from ..sql import select
from ..sql import sqltypes
from ..sql.base import _entity_namespace_key
from ..sql.base import CompileState
from ..sql.base import Options
from ..sql.dml import DeleteDMLState
from ..sql.dml import InsertDMLState
from ..sql.dml import UpdateDMLState
from ..util import EMPTY_DICT
from ..util.typing import Literal
def _setup_for_bulk_insert(self, compiler):
    """establish an INSERT statement within the context of
        bulk insert.

        This method will be within the "conn.execute()" call that is invoked
        by persistence._emit_insert_statement().

        """
    statement = orm_level_statement = cast(dml.Insert, self.statement)
    an = statement._annotations
    emit_insert_table, emit_insert_mapper = (an['_emit_insert_table'], an['_emit_insert_mapper'])
    statement = statement._clone()
    statement.table = emit_insert_table
    if self._dict_parameters:
        self._dict_parameters = {col: val for col, val in self._dict_parameters.items() if col.table is emit_insert_table}
    statement = self._setup_orm_returning(compiler, orm_level_statement, statement, dml_mapper=emit_insert_mapper, use_supplemental_cols=True)
    if self.from_statement_ctx is not None and self.from_statement_ctx.compile_options._is_star:
        raise sa_exc.CompileError("Can't use RETURNING * with bulk ORM INSERT.  Please use a different INSERT form, such as INSERT..VALUES or INSERT with a Core Connection")
    self.statement = statement