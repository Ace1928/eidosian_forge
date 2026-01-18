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
def _collect_insert_commands(table, states_to_insert, *, bulk=False, return_defaults=False, render_nulls=False, include_bulk_keys=()):
    """Identify sets of values to use in INSERT statements for a
    list of states.

    """
    for state, state_dict, mapper, connection in states_to_insert:
        if table not in mapper._pks_by_table:
            continue
        params = {}
        value_params = {}
        propkey_to_col = mapper._propkey_to_col[table]
        eval_none = mapper._insert_cols_evaluating_none[table]
        for propkey in set(propkey_to_col).intersection(state_dict):
            value = state_dict[propkey]
            col = propkey_to_col[propkey]
            if value is None and col not in eval_none and (not render_nulls):
                continue
            elif not bulk and (hasattr(value, '__clause_element__') or isinstance(value, sql.ClauseElement)):
                value_params[col] = value.__clause_element__() if hasattr(value, '__clause_element__') else value
            else:
                params[col.key] = value
        if not bulk:
            for colkey in mapper._insert_cols_as_none[table].difference(params).difference([c.key for c in value_params]):
                params[colkey] = None
        if not bulk or return_defaults:
            has_all_pks = mapper._pk_keys_by_table[table].issubset(params)
            if mapper.base_mapper._prefer_eager_defaults(connection.dialect, table):
                has_all_defaults = mapper._server_default_col_keys[table].issubset(params)
            else:
                has_all_defaults = True
        else:
            has_all_defaults = has_all_pks = True
        if mapper.version_id_generator is not False and mapper.version_id_col is not None and (mapper.version_id_col in mapper._cols_by_table[table]):
            params[mapper.version_id_col.key] = mapper.version_id_generator(None)
        if bulk:
            if mapper._set_polymorphic_identity:
                params.setdefault(mapper._polymorphic_attr_key, mapper.polymorphic_identity)
            if include_bulk_keys:
                params.update(((k, state_dict[k]) for k in include_bulk_keys))
        yield (state, state_dict, params, mapper, connection, value_params, has_all_pks, has_all_defaults)