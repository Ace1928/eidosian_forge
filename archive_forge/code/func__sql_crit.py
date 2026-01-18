import copy
from sqlalchemy import inspect
from sqlalchemy import orm
from sqlalchemy import sql
from sqlalchemy import types as sqltypes
from oslo_db.sqlalchemy import utils
def _sql_crit(expression, value):
    """Produce an equality expression against the given value.

    This takes into account a value that is actually a collection
    of values, as well as a value of None or collection that contains
    None.

    """
    values = utils.to_list(value, default=(None,))
    if len(values) == 1:
        if values[0] is None:
            return expression == sql.null()
        else:
            return expression == values[0]
    elif _none_set.intersection(values):
        return sql.or_(expression == sql.null(), _sql_crit(expression, set(values).difference(_none_set)))
    else:
        return expression.in_(values)