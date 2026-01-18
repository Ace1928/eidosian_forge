import datetime
import logging
from petl.compat import long, text_type
from petl.errors import ArgumentError
from petl.util.materialise import columns
from petl.transform.basics import head
from petl.io.db_utils import _is_dbapi_connection, _is_dbapi_cursor, \
def _execute_dbapi_cursor(sql, cursor, commit):
    debug('execute SQL')
    cursor.execute(sql)
    if commit:
        debug('commit transaction')
        assert hasattr(cursor, 'connection'), 'could not obtain connection via cursor'
        connection = cursor.connection
        connection.commit()