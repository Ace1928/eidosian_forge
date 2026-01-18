import datetime
import logging
from petl.compat import long, text_type
from petl.errors import ArgumentError
from petl.util.materialise import columns
from petl.transform.basics import head
from petl.io.db_utils import _is_dbapi_connection, _is_dbapi_cursor, \
def _execute_dbapi_connection(sql, connection, commit):
    debug('obtain a cursor')
    cursor = connection.cursor()
    debug('execute SQL')
    cursor.execute(sql)
    debug('close the cursor')
    cursor.close()
    if commit:
        debug('commit transaction')
        connection.commit()