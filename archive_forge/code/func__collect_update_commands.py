from __future__ import annotations
from itertools import chain
from itertools import groupby
from itertools import zip_longest
import operator
from . import attributes
from . import exc as orm_exc
from . import loading
from . import sync
from .base import state_str
from .. import exc as sa_exc
from .. import future
from .. import sql
from .. import util
from ..engine import cursor as _cursor
from ..sql import operators
from ..sql.elements import BooleanClauseList
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
def _collect_update_commands(uowtransaction, table, states_to_update, *, bulk=False, use_orm_update_stmt=None, include_bulk_keys=()):
    """Identify sets of values to use in UPDATE statements for a
    list of states.

    This function works intricately with the history system
    to determine exactly what values should be updated
    as well as how the row should be matched within an UPDATE
    statement.  Includes some tricky scenarios where the primary
    key of an object might have been changed.

    """
    for state, state_dict, mapper, connection, update_version_id in states_to_update:
        if table not in mapper._pks_by_table:
            continue
        pks = mapper._pks_by_table[table]
        if use_orm_update_stmt is not None:
            value_params = use_orm_update_stmt._values
        else:
            value_params = {}
        propkey_to_col = mapper._propkey_to_col[table]
        if bulk:
            params = {propkey_to_col[propkey].key: state_dict[propkey] for propkey in set(propkey_to_col).intersection(state_dict).difference(mapper._pk_attr_keys_by_table[table])}
            has_all_defaults = True
        else:
            params = {}
            for propkey in set(propkey_to_col).intersection(state.committed_state):
                value = state_dict[propkey]
                col = propkey_to_col[propkey]
                if hasattr(value, '__clause_element__') or isinstance(value, sql.ClauseElement):
                    value_params[col] = value.__clause_element__() if hasattr(value, '__clause_element__') else value
                elif state.manager[propkey].impl.is_equal(value, state.committed_state[propkey]) is not True:
                    params[col.key] = value
            if mapper.base_mapper.eager_defaults is True:
                has_all_defaults = mapper._server_onupdate_default_col_keys[table].issubset(params)
            else:
                has_all_defaults = True
        if update_version_id is not None and mapper.version_id_col in mapper._cols_by_table[table]:
            if not bulk and (not (params or value_params)):
                for prop in mapper._columntoproperty.values():
                    history = state.manager[prop.key].impl.get_history(state, state_dict, attributes.PASSIVE_NO_INITIALIZE)
                    if history.added:
                        break
                else:
                    continue
            col = mapper.version_id_col
            no_params = not params and (not value_params)
            params[col._label] = update_version_id
            if (bulk or col.key not in params) and mapper.version_id_generator is not False:
                val = mapper.version_id_generator(update_version_id)
                params[col.key] = val
            elif mapper.version_id_generator is False and no_params:
                params[col.key] = update_version_id
        elif not (params or value_params):
            continue
        has_all_pks = True
        expect_pk_cascaded = False
        if bulk:
            pk_params = {propkey_to_col[propkey]._label: state_dict.get(propkey) for propkey in set(propkey_to_col).intersection(mapper._pk_attr_keys_by_table[table])}
            if util.NONE_SET.intersection(pk_params.values()):
                raise sa_exc.InvalidRequestError(f'No primary key value supplied for column(s) {', '.join((str(c) for c in pks if pk_params[c._label] is None))}; per-row ORM Bulk UPDATE by Primary Key requires that records contain primary key values', code='bupq')
        else:
            pk_params = {}
            for col in pks:
                propkey = mapper._columntoproperty[col].key
                history = state.manager[propkey].impl.get_history(state, state_dict, attributes.PASSIVE_OFF)
                if history.added:
                    if not history.deleted or ('pk_cascaded', state, col) in uowtransaction.attributes:
                        expect_pk_cascaded = True
                        pk_params[col._label] = history.added[0]
                        params.pop(col.key, None)
                    else:
                        pk_params[col._label] = history.deleted[0]
                        if col in value_params:
                            has_all_pks = False
                else:
                    pk_params[col._label] = history.unchanged[0]
                if pk_params[col._label] is None:
                    raise orm_exc.FlushError("Can't update table %s using NULL for primary key value on column %s" % (table, col))
        if include_bulk_keys:
            params.update(((k, state_dict[k]) for k in include_bulk_keys))
        if params or value_params:
            params.update(pk_params)
            yield (state, state_dict, params, mapper, connection, value_params, has_all_defaults, has_all_pks)
        elif expect_pk_cascaded:
            for m, equated_pairs in mapper._table_to_equated[table]:
                sync.populate(state, m, state, m, equated_pairs, uowtransaction, mapper.passive_updates)