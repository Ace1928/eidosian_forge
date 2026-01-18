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
@CompileState.plugin_for('orm', 'update')
class BulkORMUpdate(BulkUDCompileState, UpdateDMLState):

    @classmethod
    def create_for_statement(cls, statement, compiler, **kw):
        self = cls.__new__(cls)
        dml_strategy = statement._annotations.get('dml_strategy', 'unspecified')
        toplevel = not compiler.stack
        if toplevel and dml_strategy == 'bulk':
            self._setup_for_bulk_update(statement, compiler)
        elif dml_strategy == 'core_only' or (dml_strategy == 'unspecified' and 'parententity' not in statement.table._annotations):
            UpdateDMLState.__init__(self, statement, compiler, **kw)
        elif not toplevel or dml_strategy in ('orm', 'unspecified'):
            self._setup_for_orm_update(statement, compiler)
        return self

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

    def _setup_for_bulk_update(self, statement, compiler, **kw):
        """establish an UPDATE statement within the context of
        bulk insert.

        This method will be within the "conn.execute()" call that is invoked
        by persistence._emit_update_statement().

        """
        statement = cast(dml.Update, statement)
        an = statement._annotations
        emit_update_table, _ = (an['_emit_update_table'], an['_emit_update_mapper'])
        statement = statement._clone()
        statement.table = emit_update_table
        UpdateDMLState.__init__(self, statement, compiler, **kw)
        if self._ordered_values:
            raise sa_exc.InvalidRequestError('bulk ORM UPDATE does not support ordered_values() for custom UPDATE statements with bulk parameter sets.  Use a non-bulk UPDATE statement or use values().')
        if self._dict_parameters:
            self._dict_parameters = {col: val for col, val in self._dict_parameters.items() if col.table is emit_update_table}
        self.statement = statement

    @classmethod
    def orm_execute_statement(cls, session: Session, statement: dml.Update, params: _CoreAnyExecuteParams, execution_options: OrmExecuteOptionsParameter, bind_arguments: _BindArguments, conn: Connection) -> _result.Result:
        update_options = execution_options.get('_sa_orm_update_options', cls.default_update_options)
        if update_options._dml_strategy not in ('orm', 'auto', 'bulk', 'core_only'):
            raise sa_exc.ArgumentError("Valid strategies for ORM UPDATE strategy are 'orm', 'auto', 'bulk', 'core_only'")
        result: _result.Result[Any]
        if update_options._dml_strategy == 'bulk':
            enable_check_rowcount = not statement._where_criteria
            assert update_options._synchronize_session != 'fetch'
            if statement._where_criteria and update_options._synchronize_session == 'evaluate':
                raise sa_exc.InvalidRequestError('bulk synchronize of persistent objects not supported when using bulk update with additional WHERE criteria right now.  add synchronize_session=None execution option to bypass synchronize of persistent objects.')
            mapper = update_options._subject_mapper
            assert mapper is not None
            assert session._transaction is not None
            result = _bulk_update(mapper, cast('Iterable[Dict[str, Any]]', [params] if isinstance(params, dict) else params), session._transaction, isstates=False, update_changed_only=False, use_orm_update_stmt=statement, enable_check_rowcount=enable_check_rowcount)
            return cls.orm_setup_cursor_result(session, statement, params, execution_options, bind_arguments, result)
        else:
            return super().orm_execute_statement(session, statement, params, execution_options, bind_arguments, conn)

    @classmethod
    def can_use_returning(cls, dialect: Dialect, mapper: Mapper[Any], *, is_multitable: bool=False, is_update_from: bool=False, is_delete_using: bool=False, is_executemany: bool=False) -> bool:
        normal_answer = dialect.update_returning and mapper.local_table.implicit_returning
        if not normal_answer:
            return False
        if is_executemany:
            return dialect.update_executemany_returning
        if is_update_from:
            return dialect.update_returning_multifrom
        elif is_multitable and (not dialect.update_returning_multifrom):
            raise sa_exc.CompileError(f'''Dialect "{dialect.name}" does not support RETURNING with UPDATE..FROM; for synchronize_session='fetch', please add the additional execution option 'is_update_from=True' to the statement to indicate that a separate SELECT should be used for this backend.''')
        return True

    @classmethod
    def _do_post_synchronize_bulk_evaluate(cls, session, params, result, update_options):
        if not params:
            return
        mapper = update_options._subject_mapper
        pk_keys = [prop.key for prop in mapper._identity_key_props]
        identity_map = session.identity_map
        for param in params:
            identity_key = mapper.identity_key_from_primary_key((param[key] for key in pk_keys), update_options._identity_token)
            state = identity_map.fast_get_state(identity_key)
            if not state:
                continue
            evaluated_keys = set(param).difference(pk_keys)
            dict_ = state.dict
            to_evaluate = state.unmodified.intersection(evaluated_keys)
            for key in to_evaluate:
                if key in dict_:
                    dict_[key] = param[key]
            state.manager.dispatch.refresh(state, None, to_evaluate)
            state._commit(dict_, list(to_evaluate))
            to_expire = evaluated_keys.intersection(dict_).difference(to_evaluate)
            if to_expire:
                state._expire_attributes(dict_, to_expire)

    @classmethod
    def _do_post_synchronize_evaluate(cls, session, statement, result, update_options):
        matched_objects = cls._get_matched_objects_on_criteria(update_options, session.identity_map.all_states())
        cls._apply_update_set_values_to_objects(session, update_options, statement, [(obj, state, dict_) for obj, state, dict_, _ in matched_objects])

    @classmethod
    def _do_post_synchronize_fetch(cls, session, statement, result, update_options):
        target_mapper = update_options._subject_mapper
        returned_defaults_rows = result.returned_defaults_rows
        if returned_defaults_rows:
            pk_rows = cls._interpret_returning_rows(target_mapper, returned_defaults_rows)
            matched_rows = [tuple(row) + (update_options._identity_token,) for row in pk_rows]
        else:
            matched_rows = update_options._matched_rows
        objs = [session.identity_map[identity_key] for identity_key in [target_mapper.identity_key_from_primary_key(list(primary_key), identity_token=identity_token) for primary_key, identity_token in [(row[0:-1], row[-1]) for row in matched_rows] if update_options._identity_token is None or identity_token == update_options._identity_token] if identity_key in session.identity_map]
        if not objs:
            return
        cls._apply_update_set_values_to_objects(session, update_options, statement, [(obj, attributes.instance_state(obj), attributes.instance_dict(obj)) for obj in objs])

    @classmethod
    def _apply_update_set_values_to_objects(cls, session, update_options, statement, matched_objects):
        """apply values to objects derived from an update statement, e.g.
        UPDATE..SET <values>

        """
        mapper = update_options._subject_mapper
        target_cls = mapper.class_
        evaluator_compiler = evaluator._EvaluatorCompiler(target_cls)
        resolved_values = cls._get_resolved_values(mapper, statement)
        resolved_keys_as_propnames = cls._resolved_keys_as_propnames(mapper, resolved_values)
        value_evaluators = {}
        for key, value in resolved_keys_as_propnames:
            try:
                _evaluator = evaluator_compiler.process(coercions.expect(roles.ExpressionElementRole, value))
            except evaluator.UnevaluatableError:
                pass
            else:
                value_evaluators[key] = _evaluator
        evaluated_keys = list(value_evaluators.keys())
        attrib = {k for k, v in resolved_keys_as_propnames}
        states = set()
        for obj, state, dict_ in matched_objects:
            to_evaluate = state.unmodified.intersection(evaluated_keys)
            for key in to_evaluate:
                if key in dict_:
                    dict_[key] = value_evaluators[key](obj)
            state.manager.dispatch.refresh(state, None, to_evaluate)
            state._commit(dict_, list(to_evaluate))
            to_expire = attrib.intersection(dict_).difference(to_evaluate)
            if to_expire:
                state._expire_attributes(dict_, to_expire)
            states.add(state)
        session._register_altered(states)