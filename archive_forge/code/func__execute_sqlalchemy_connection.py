import datetime
import logging
from petl.compat import long, text_type
from petl.errors import ArgumentError
from petl.util.materialise import columns
from petl.transform.basics import head
from petl.io.db_utils import _is_dbapi_connection, _is_dbapi_cursor, \
def _execute_sqlalchemy_connection(sql, connection, commit):
    if commit:
        debug('begin transaction')
        trans = connection.begin()
    debug('execute SQL')
    connection.execute(sql)
    if commit:
        debug('commit transaction')
        trans.commit()