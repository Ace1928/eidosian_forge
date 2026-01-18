from __future__ import absolute_import, print_function, division
import logging
from petl.compat import next, text_type, string_types
from petl.errors import ArgumentError
from petl.util.base import Table
from petl.io.db_utils import _is_dbapi_connection, _is_dbapi_cursor, \
from petl.io.db_create import drop_table, create_table
class DbView(Table):

    def __init__(self, dbo, query, *args, **kwargs):
        self.dbo = dbo
        self.query = query
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        if _is_dbapi_connection(self.dbo):
            debug('assuming %r is standard DB-API 2.0 connection', self.dbo)
            _iter = _iter_dbapi_connection
        elif _is_dbapi_cursor(self.dbo):
            debug('assuming %r is standard DB-API 2.0 cursor')
            warning('using a DB-API cursor with fromdb() is not recommended and may lead to unexpected results, a DB-API connection is better')
            _iter = _iter_dbapi_cursor
        elif _is_sqlalchemy_engine(self.dbo):
            debug('assuming %r instance of sqlalchemy.engine.base.Engine', self.dbo)
            _iter = _iter_sqlalchemy_engine
        elif _is_sqlalchemy_session(self.dbo):
            debug('assuming %r instance of sqlalchemy.orm.session.Session', self.dbo)
            _iter = _iter_sqlalchemy_session
        elif _is_sqlalchemy_connection(self.dbo):
            debug('assuming %r instance of sqlalchemy.engine.base.Connection', self.dbo)
            _iter = _iter_sqlalchemy_connection
        elif callable(self.dbo):
            debug('assuming %r is a function returning a cursor', self.dbo)
            _iter = _iter_dbapi_mkcurs
        else:
            raise ArgumentError('unsupported database object type: %r' % self.dbo)
        return _iter(self.dbo, self.query, *self.args, **self.kwargs)