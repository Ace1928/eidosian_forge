from __future__ import absolute_import, print_function, division
import logging
from petl.compat import next, text_type, string_types
from petl.errors import ArgumentError
from petl.util.base import Table
from petl.io.db_utils import _is_dbapi_connection, _is_dbapi_cursor, \
from petl.io.db_create import drop_table, create_table
def _iter_sqlalchemy_connection(connection, query, *args, **kwargs):
    debug('connection: %r', connection)
    results = connection.execute(query, *args, **kwargs)
    hdr = results.keys()
    yield tuple(hdr)
    for row in results:
        yield row