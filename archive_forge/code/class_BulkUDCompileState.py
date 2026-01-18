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
class BulkUDCompileState(ORMDMLState):

    class default_update_options(Options):
        _dml_strategy: DMLStrategyArgument = 'auto'
        _synchronize_session: SynchronizeSessionArgument = 'auto'
        _can_use_returning: bool = False
        _is_delete_using: bool = False
        _is_update_from: bool = False
        _autoflush: bool = True
        _subject_mapper: Optional[Mapper[Any]] = None
        _resolved_values = EMPTY_DICT
        _eval_condition = None
        _matched_rows = None
        _identity_token = None

    @classmethod
    def can_use_returning(cls, dialect: Dialect, mapper: Mapper[Any], *, is_multitable: bool=False, is_update_from: bool=False, is_delete_using: bool=False, is_executemany: bool=False) -> bool:
        raise NotImplementedError()

    @classmethod
    def orm_pre_session_exec(cls, session, statement, params, execution_options, bind_arguments, is_pre_event):
        update_options, execution_options = BulkUDCompileState.default_update_options.from_execution_options('_sa_orm_update_options', {'synchronize_session', 'autoflush', 'identity_token', 'is_delete_using', 'is_update_from', 'dml_strategy'}, execution_options, statement._execution_options)
        bind_arguments['clause'] = statement
        try:
            plugin_subject = statement._propagate_attrs['plugin_subject']
        except KeyError:
            assert False, "statement had 'orm' plugin but no plugin_subject"
        else:
            if plugin_subject:
                bind_arguments['mapper'] = plugin_subject.mapper
                update_options += {'_subject_mapper': plugin_subject.mapper}
        if 'parententity' not in statement.table._annotations:
            update_options += {'_dml_strategy': 'core_only'}
        elif not isinstance(params, list):
            if update_options._dml_strategy == 'auto':
                update_options += {'_dml_strategy': 'orm'}
            elif update_options._dml_strategy == 'bulk':
                raise sa_exc.InvalidRequestError('Can\'t use "bulk" ORM insert strategy without passing separate parameters')
        elif update_options._dml_strategy == 'auto':
            update_options += {'_dml_strategy': 'bulk'}
        sync = update_options._synchronize_session
        if sync is not None:
            if sync not in ('auto', 'evaluate', 'fetch', False):
                raise sa_exc.ArgumentError("Valid strategies for session synchronization are 'auto', 'evaluate', 'fetch', False")
            if update_options._dml_strategy == 'bulk' and sync == 'fetch':
                raise sa_exc.InvalidRequestError("The 'fetch' synchronization strategy is not available for 'bulk' ORM updates (i.e. multiple parameter sets)")
        if not is_pre_event:
            if update_options._autoflush:
                session._autoflush()
            if update_options._dml_strategy == 'orm':
                if update_options._synchronize_session == 'auto':
                    update_options = cls._do_pre_synchronize_auto(session, statement, params, execution_options, bind_arguments, update_options)
                elif update_options._synchronize_session == 'evaluate':
                    update_options = cls._do_pre_synchronize_evaluate(session, statement, params, execution_options, bind_arguments, update_options)
                elif update_options._synchronize_session == 'fetch':
                    update_options = cls._do_pre_synchronize_fetch(session, statement, params, execution_options, bind_arguments, update_options)
            elif update_options._dml_strategy == 'bulk':
                if update_options._synchronize_session == 'auto':
                    update_options += {'_synchronize_session': 'evaluate'}
            statement = statement._annotate({'synchronize_session': update_options._synchronize_session, 'is_delete_using': update_options._is_delete_using, 'is_update_from': update_options._is_update_from, 'dml_strategy': update_options._dml_strategy, 'can_use_returning': update_options._can_use_returning})
        return (statement, util.immutabledict(execution_options).union({'_sa_orm_update_options': update_options}))

    @classmethod
    def orm_setup_cursor_result(cls, session, statement, params, execution_options, bind_arguments, result):
        update_options = execution_options['_sa_orm_update_options']
        if update_options._dml_strategy == 'orm':
            if update_options._synchronize_session == 'evaluate':
                cls._do_post_synchronize_evaluate(session, statement, result, update_options)
            elif update_options._synchronize_session == 'fetch':
                cls._do_post_synchronize_fetch(session, statement, result, update_options)
        elif update_options._dml_strategy == 'bulk':
            if update_options._synchronize_session == 'evaluate':
                cls._do_post_synchronize_bulk_evaluate(session, params, result, update_options)
            return result
        return cls._return_orm_returning(session, statement, params, execution_options, bind_arguments, result)

    @classmethod
    def _adjust_for_extra_criteria(cls, global_attributes, ext_info):
        """Apply extra criteria filtering.

        For all distinct single-table-inheritance mappers represented in the
        table being updated or deleted, produce additional WHERE criteria such
        that only the appropriate subtypes are selected from the total results.

        Additionally, add WHERE criteria originating from LoaderCriteriaOptions
        collected from the statement.

        """
        return_crit = ()
        adapter = ext_info._adapter if ext_info.is_aliased_class else None
        if ('additional_entity_criteria', ext_info.mapper) in global_attributes:
            return_crit += tuple((ae._resolve_where_criteria(ext_info) for ae in global_attributes['additional_entity_criteria', ext_info.mapper] if ae.include_aliases or ae.entity is ext_info))
        if ext_info.mapper._single_table_criterion is not None:
            return_crit += (ext_info.mapper._single_table_criterion,)
        if adapter:
            return_crit = tuple((adapter.traverse(crit) for crit in return_crit))
        return return_crit

    @classmethod
    def _interpret_returning_rows(cls, mapper, rows):
        """translate from local inherited table columns to base mapper
        primary key columns.

        Joined inheritance mappers always establish the primary key in terms of
        the base table.   When we UPDATE a sub-table, we can only get
        RETURNING for the sub-table's columns.

        Here, we create a lookup from the local sub table's primary key
        columns to the base table PK columns so that we can get identity
        key values from RETURNING that's against the joined inheritance
        sub-table.

        the complexity here is to support more than one level deep of
        inheritance, where we have to link columns to each other across
        the inheritance hierarchy.

        """
        if mapper.local_table is not mapper.base_mapper.local_table:
            return rows
        local_pk_to_base_pk = {pk: pk for pk in mapper.local_table.primary_key}
        for mp in mapper.iterate_to_root():
            if mp.inherits is None:
                break
            elif mp.local_table is mp.inherits.local_table:
                continue
            t_to_e = dict(mp._table_to_equated[mp.inherits.local_table])
            col_to_col = {sub_pk: super_pk for super_pk, sub_pk in t_to_e[mp]}
            for pk, super_ in local_pk_to_base_pk.items():
                local_pk_to_base_pk[pk] = col_to_col[super_]
        lookup = {local_pk_to_base_pk[lpk]: idx for idx, lpk in enumerate(mapper.local_table.primary_key)}
        primary_key_convert = [lookup[bpk] for bpk in mapper.base_mapper.primary_key]
        return [tuple((row[idx] for idx in primary_key_convert)) for row in rows]

    @classmethod
    def _get_matched_objects_on_criteria(cls, update_options, states):
        mapper = update_options._subject_mapper
        eval_condition = update_options._eval_condition
        raw_data = [(state.obj(), state, state.dict) for state in states if state.mapper.isa(mapper) and (not state.expired)]
        identity_token = update_options._identity_token
        if identity_token is not None:
            raw_data = [(obj, state, dict_) for obj, state, dict_ in raw_data if state.identity_token == identity_token]
        result = []
        for obj, state, dict_ in raw_data:
            evaled_condition = eval_condition(obj)
            if evaled_condition is True or evaled_condition is evaluator._EXPIRED_OBJECT:
                result.append((obj, state, dict_, evaled_condition is evaluator._EXPIRED_OBJECT))
        return result

    @classmethod
    def _eval_condition_from_statement(cls, update_options, statement):
        mapper = update_options._subject_mapper
        target_cls = mapper.class_
        evaluator_compiler = evaluator._EvaluatorCompiler(target_cls)
        crit = ()
        if statement._where_criteria:
            crit += statement._where_criteria
        global_attributes = {}
        for opt in statement._with_options:
            if opt._is_criteria_option:
                opt.get_global_criteria(global_attributes)
        if global_attributes:
            crit += cls._adjust_for_extra_criteria(global_attributes, mapper)
        if crit:
            eval_condition = evaluator_compiler.process(*crit)
        else:

            def _eval_condition(obj):
                return True
            eval_condition = _eval_condition
        return eval_condition

    @classmethod
    def _do_pre_synchronize_auto(cls, session, statement, params, execution_options, bind_arguments, update_options):
        """setup auto sync strategy


        "auto" checks if we can use "evaluate" first, then falls back
        to "fetch"

        evaluate is vastly more efficient for the common case
        where session is empty, only has a few objects, and the UPDATE
        statement can potentially match thousands/millions of rows.

        OTOH more complex criteria that fails to work with "evaluate"
        we would hope usually correlates with fewer net rows.

        """
        try:
            eval_condition = cls._eval_condition_from_statement(update_options, statement)
        except evaluator.UnevaluatableError:
            pass
        else:
            return update_options + {'_eval_condition': eval_condition, '_synchronize_session': 'evaluate'}
        update_options += {'_synchronize_session': 'fetch'}
        return cls._do_pre_synchronize_fetch(session, statement, params, execution_options, bind_arguments, update_options)

    @classmethod
    def _do_pre_synchronize_evaluate(cls, session, statement, params, execution_options, bind_arguments, update_options):
        try:
            eval_condition = cls._eval_condition_from_statement(update_options, statement)
        except evaluator.UnevaluatableError as err:
            raise sa_exc.InvalidRequestError('Could not evaluate current criteria in Python: "%s". Specify \'fetch\' or False for the synchronize_session execution option.' % err) from err
        return update_options + {'_eval_condition': eval_condition}

    @classmethod
    def _get_resolved_values(cls, mapper, statement):
        if statement._multi_values:
            return []
        elif statement._ordered_values:
            return list(statement._ordered_values)
        elif statement._values:
            return list(statement._values.items())
        else:
            return []

    @classmethod
    def _resolved_keys_as_propnames(cls, mapper, resolved_values):
        values = []
        for k, v in resolved_values:
            if mapper and isinstance(k, expression.ColumnElement):
                try:
                    attr = mapper._columntoproperty[k]
                except orm_exc.UnmappedColumnError:
                    pass
                else:
                    values.append((attr.key, v))
            else:
                raise sa_exc.InvalidRequestError("Attribute name not found, can't be synchronized back to objects: %r" % k)
        return values

    @classmethod
    def _do_pre_synchronize_fetch(cls, session, statement, params, execution_options, bind_arguments, update_options):
        mapper = update_options._subject_mapper
        select_stmt = select(*mapper.primary_key + (mapper.select_identity_token,)).select_from(mapper).options(*statement._with_options)
        select_stmt._where_criteria = statement._where_criteria
        can_use_returning = None

        def skip_for_returning(orm_context: ORMExecuteState) -> Any:
            bind = orm_context.session.get_bind(**orm_context.bind_arguments)
            nonlocal can_use_returning
            per_bind_result = cls.can_use_returning(bind.dialect, mapper, is_update_from=update_options._is_update_from, is_delete_using=update_options._is_delete_using, is_executemany=orm_context.is_executemany)
            if can_use_returning is not None:
                if can_use_returning != per_bind_result:
                    raise sa_exc.InvalidRequestError("For synchronize_session='fetch', can't mix multiple backends where some support RETURNING and others don't")
            elif orm_context.is_executemany and (not per_bind_result):
                raise sa_exc.InvalidRequestError("For synchronize_session='fetch', can't use multiple parameter sets in ORM mode, which this backend does not support with RETURNING")
            else:
                can_use_returning = per_bind_result
            if per_bind_result:
                return _result.null_result()
            else:
                return None
        result = session.execute(select_stmt, params, execution_options=execution_options, bind_arguments=bind_arguments, _add_event=skip_for_returning)
        matched_rows = result.fetchall()
        return update_options + {'_matched_rows': matched_rows, '_can_use_returning': can_use_returning}