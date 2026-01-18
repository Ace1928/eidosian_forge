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
def _setup_for_orm_update(self, statement, compiler, **kw):
    orm_level_statement = statement
    toplevel = not compiler.stack
    ext_info = statement.table._annotations['parententity']
    self.mapper = mapper = ext_info.mapper
    self._resolved_values = self._get_resolved_values(mapper, statement)
    self._init_global_attributes(statement, compiler, toplevel=toplevel, process_criteria_for_toplevel=toplevel)
    if statement._values:
        self._resolved_values = dict(self._resolved_values)
    new_stmt = statement._clone()
    if statement._ordered_values:
        new_stmt._ordered_values = self._resolved_values
    elif statement._values:
        new_stmt._values = self._resolved_values
    new_crit = self._adjust_for_extra_criteria(self.global_attributes, mapper)
    if new_crit:
        new_stmt = new_stmt.where(*new_crit)
    UpdateDMLState.__init__(self, new_stmt, compiler, **kw)
    use_supplemental_cols = False
    if not toplevel:
        synchronize_session = None
    else:
        synchronize_session = compiler._annotations.get('synchronize_session', None)
    can_use_returning = compiler._annotations.get('can_use_returning', None)
    if can_use_returning is not False:
        can_use_returning = synchronize_session == 'fetch' and self.can_use_returning(compiler.dialect, mapper, is_multitable=self.is_multitable)
    if synchronize_session == 'fetch' and can_use_returning:
        use_supplemental_cols = True
        new_stmt = new_stmt.return_defaults(*new_stmt.table.primary_key)
    if toplevel:
        new_stmt = self._setup_orm_returning(compiler, orm_level_statement, new_stmt, dml_mapper=mapper, use_supplemental_cols=use_supplemental_cols)
    self.statement = new_stmt