import json
import logging
import uuid
from peewee import *
from peewee import ColumnBase
from peewee import Expression
from peewee import Node
from peewee import NodeList
from peewee import __deprecated__
class PostgresqlExtDatabase(PostgresqlDatabase):

    def __init__(self, *args, **kwargs):
        self._register_hstore = kwargs.pop('register_hstore', False)
        self._server_side_cursors = kwargs.pop('server_side_cursors', False)
        super(PostgresqlExtDatabase, self).__init__(*args, **kwargs)

    def _connect(self):
        conn = super(PostgresqlExtDatabase, self)._connect()
        if self._register_hstore:
            register_hstore(conn, globally=True)
        return conn

    def cursor(self, commit=None, named_cursor=None):
        if commit is not None:
            __deprecated__('"commit" has been deprecated and is a no-op.')
        if self.is_closed():
            if self.autoconnect:
                self.connect()
            else:
                raise InterfaceError('Error, database connection not opened.')
        if named_cursor:
            curs = self._state.conn.cursor(name=str(uuid.uuid1()))
            return curs
        return self._state.conn.cursor()

    def execute(self, query, commit=None, named_cursor=False, array_size=None, **context_options):
        if commit is not None:
            __deprecated__('"commit" has been deprecated and is a no-op.')
        ctx = self.get_sql_context(**context_options)
        sql, params = ctx.sql(query).query()
        named_cursor = named_cursor or (self._server_side_cursors and sql[:6].lower() == 'select')
        cursor = self.execute_sql(sql, params)
        if named_cursor:
            cursor = FetchManyCursor(cursor, array_size)
        return cursor