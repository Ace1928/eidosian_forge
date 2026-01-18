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
@CompileState.plugin_for('orm', 'delete')
class BulkORMDelete(BulkUDCompileState, DeleteDMLState):

    @classmethod
    def create_for_statement(cls, statement, compiler, **kw):
        self = cls.__new__(cls)
        dml_strategy = statement._annotations.get('dml_strategy', 'unspecified')
        if dml_strategy == 'core_only' or (dml_strategy == 'unspecified' and 'parententity' not in statement.table._annotations):
            DeleteDMLState.__init__(self, statement, compiler, **kw)
            return self
        toplevel = not compiler.stack
        orm_level_statement = statement
        ext_info = statement.table._annotations['parententity']
        self.mapper = mapper = ext_info.mapper
        self._init_global_attributes(statement, compiler, toplevel=toplevel, process_criteria_for_toplevel=toplevel)
        new_stmt = statement._clone()
        new_crit = cls._adjust_for_extra_criteria(self.global_attributes, mapper)
        if new_crit:
            new_stmt = new_stmt.where(*new_crit)
        DeleteDMLState.__init__(self, new_stmt, compiler, **kw)
        use_supplemental_cols = False
        if not toplevel:
            synchronize_session = None
        else:
            synchronize_session = compiler._annotations.get('synchronize_session', None)
        can_use_returning = compiler._annotations.get('can_use_returning', None)
        if can_use_returning is not False:
            can_use_returning = synchronize_session == 'fetch' and self.can_use_returning(compiler.dialect, mapper, is_multitable=self.is_multitable, is_delete_using=compiler._annotations.get('is_delete_using', False))
        if can_use_returning:
            use_supplemental_cols = True
            new_stmt = new_stmt.return_defaults(*new_stmt.table.primary_key)
        if toplevel:
            new_stmt = self._setup_orm_returning(compiler, orm_level_statement, new_stmt, dml_mapper=mapper, use_supplemental_cols=use_supplemental_cols)
        self.statement = new_stmt
        return self

    @classmethod
    def orm_execute_statement(cls, session: Session, statement: dml.Delete, params: _CoreAnyExecuteParams, execution_options: OrmExecuteOptionsParameter, bind_arguments: _BindArguments, conn: Connection) -> _result.Result:
        update_options = execution_options.get('_sa_orm_update_options', cls.default_update_options)
        if update_options._dml_strategy == 'bulk':
            raise sa_exc.InvalidRequestError('Bulk ORM DELETE not supported right now. Statement may be invoked at the Core level using session.connection().execute(stmt, parameters)')
        if update_options._dml_strategy not in ('orm', 'auto', 'core_only'):
            raise sa_exc.ArgumentError("Valid strategies for ORM DELETE strategy are 'orm', 'auto', 'core_only'")
        return super().orm_execute_statement(session, statement, params, execution_options, bind_arguments, conn)

    @classmethod
    def can_use_returning(cls, dialect: Dialect, mapper: Mapper[Any], *, is_multitable: bool=False, is_update_from: bool=False, is_delete_using: bool=False, is_executemany: bool=False) -> bool:
        normal_answer = dialect.delete_returning and mapper.local_table.implicit_returning
        if not normal_answer:
            return False
        if is_delete_using:
            return dialect.delete_returning_multifrom
        elif is_multitable and (not dialect.delete_returning_multifrom):
            raise sa_exc.CompileError(f'''Dialect "{dialect.name}" does not support RETURNING with DELETE..USING; for synchronize_session='fetch', please add the additional execution option 'is_delete_using=True' to the statement to indicate that a separate SELECT should be used for this backend.''')
        return True

    @classmethod
    def _do_post_synchronize_evaluate(cls, session, statement, result, update_options):
        matched_objects = cls._get_matched_objects_on_criteria(update_options, session.identity_map.all_states())
        to_delete = []
        for _, state, dict_, is_partially_expired in matched_objects:
            if is_partially_expired:
                state._expire(dict_, session.identity_map._modified)
            else:
                to_delete.append(state)
        if to_delete:
            session._remove_newly_deleted(to_delete)

    @classmethod
    def _do_post_synchronize_fetch(cls, session, statement, result, update_options):
        target_mapper = update_options._subject_mapper
        returned_defaults_rows = result.returned_defaults_rows
        if returned_defaults_rows:
            pk_rows = cls._interpret_returning_rows(target_mapper, returned_defaults_rows)
            matched_rows = [tuple(row) + (update_options._identity_token,) for row in pk_rows]
        else:
            matched_rows = update_options._matched_rows
        for row in matched_rows:
            primary_key = row[0:-1]
            identity_token = row[-1]
            identity_key = target_mapper.identity_key_from_primary_key(list(primary_key), identity_token=identity_token)
            if identity_key in session.identity_map:
                session._remove_newly_deleted([attributes.instance_state(session.identity_map[identity_key])])