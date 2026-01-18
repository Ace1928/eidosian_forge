from __future__ import absolute_import, print_function, division
import logging
from petl.compat import next, text_type, string_types
from petl.errors import ArgumentError
from petl.util.base import Table
from petl.io.db_utils import _is_dbapi_connection, _is_dbapi_cursor, \
from petl.io.db_create import drop_table, create_table
def _iter_dbapi_cursor(cursor, query, *args, **kwargs):
    cursor.execute(query, *args, **kwargs)
    try:
        it = iter(cursor)
    except TypeError:
        it = iter(cursor.fetchall())
    try:
        first_row = next(it)
    except StopIteration:
        first_row = None
    hdr = [d[0] for d in cursor.description]
    yield tuple(hdr)
    if first_row is None:
        return
    yield first_row
    for row in it:
        yield row