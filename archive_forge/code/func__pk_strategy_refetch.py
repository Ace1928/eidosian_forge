import copy
from sqlalchemy import inspect
from sqlalchemy import orm
from sqlalchemy import sql
from sqlalchemy import types as sqltypes
from oslo_db.sqlalchemy import utils
def _pk_strategy_refetch(query, mapper, values, surrogate_key):
    surrogate_key_name, surrogate_key_value = surrogate_key
    surrogate_key_col = mapper.attrs[surrogate_key_name].expression
    rowcount = query.filter(surrogate_key_col == surrogate_key_value).update(values, synchronize_session=False)
    _assert_single_row(rowcount)
    fetch_query = query.session.query(*mapper.primary_key).filter(surrogate_key_col == surrogate_key_value)
    primary_key = fetch_query.one()
    return primary_key