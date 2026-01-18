import copy
from sqlalchemy import inspect
from sqlalchemy import orm
from sqlalchemy import sql
from sqlalchemy import types as sqltypes
from oslo_db.sqlalchemy import utils
def _assert_single_row(rows_updated):
    if rows_updated == 1:
        return rows_updated
    elif rows_updated > 1:
        raise MultiRowsMatched('%d rows matched; expected one' % rows_updated)
    else:
        raise NoRowsMatched('No rows matched the UPDATE')